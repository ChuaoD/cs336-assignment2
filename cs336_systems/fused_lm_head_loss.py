from dataclasses import dataclass
from jaxtyping import Float, Int
import math
import torch
from torch import Tensor
import triton
import triton.language as tl


def fused_lm_head_loss_forward_torch(
    hidden: Float[Tensor, "Nt D"],
    weight: Float[Tensor, "V D"],
    targets: Int[Tensor, "Nt"],
    Bt: int,
    Bv: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (scalar_loss, logsumexp) without full logits.

    Shapes:
        scalar_loss: []
        logsumexp: [Nt], FP32
    """
    Nt = hidden.shape[0]
    Tt = math.ceil(Nt / Bt)
    V = weight.shape[0]
    Tv = math.ceil(V / Bv)
    tiled_loss = torch.empty(Nt, dtype=torch.float32, device=hidden.device)
    lse = torch.empty(Nt, dtype=torch.float32, device=hidden.device)
    for i in range(Tt):
        h_start = i * Bt
        h_end = min(h_start + Bt, Nt)
        h_tile_size = h_end - h_start
        h_tile = hidden[h_start:h_end, :]  # (Bt, D)
        m_old = torch.full(
            (h_tile_size,),
            float('-inf'),
            device=hidden.device,
            dtype=torch.float32 
        ) 
        l = torch.zeros(h_tile_size, device=hidden.device, dtype=torch.float32)
        targets_tile = targets[h_start:h_end]  # (Bt,)
        targets_z_tile = torch.zeros(
            h_tile_size,
            device=hidden.device,
            dtype=torch.float32,
        )
        for j in range(Tv):
            w_start = j * Bv
            w_end = min(w_start + Bv, V)
            w_tile_size = w_end - w_start
            w_tile = weight[w_start:w_end, :]  # (Bv, D)
            z_tile = torch.matmul(h_tile, w_tile.transpose(-2, -1))  # (Bt, Bv)
            m = torch.maximum(m_old, z_tile.amax(dim=-1))
            l = torch.exp(m_old - m) * l + torch.exp(z_tile - m.unsqueeze(-1)).sum(dim=-1)
            m_old = m

            target_in_tile = (targets_tile >= w_start) & (targets_tile < w_end)
            local_target = (targets_tile - w_start).clamp(0, w_tile_size - 1)
            candidate_target_z = torch.gather(
                z_tile,
                dim=-1,
                index=local_target.unsqueeze(-1),
            ).squeeze(-1)
            targets_z_tile = torch.where(
                target_in_tile,
                candidate_target_z,
                targets_z_tile,
            )
        lse_tile = m + torch.log(l)
        lse[h_start:h_end] = lse_tile
        loss_tile = lse_tile - targets_z_tile
        tiled_loss[h_start:h_end] = loss_tile
    return tiled_loss.sum(), lse


@triton.jit
def fused_lm_head_loss_forward_kernel(
    H_ptr,
    W_ptr,
    targets_ptr,
    partial_loss_sums_ptr,
    lse_ptr,
    stride_ht, stride_hd,
    stride_wv, stride_wd,
    stride_t,
    stride_pls,
    stride_lse,
    Nt,
    Kt,
    D: tl.constexpr,
    V: tl.constexpr,
    H_TILE_SIZE: tl.constexpr,
    W_TILE_SIZE: tl.constexpr,
    D_TILE_SIZE: tl.constexpr
):
    """Write FP32 token losses and log-sum-exp values for one token tile."""
    # Program indices
    H_tile_index = tl.program_id(0)
    targets_block_ptr = tl.make_block_ptr(
        targets_ptr,
        shape=(Nt,),
        strides=(stride_t,),
        offsets=(H_tile_index * H_TILE_SIZE,),
        block_shape=(H_TILE_SIZE,),
        order=(0,)
    )
    partial_loss_sums_block_ptr = tl.make_block_ptr(
        partial_loss_sums_ptr,
        shape=(Kt,),
        strides=(stride_pls,),
        offsets=(H_tile_index,),
        block_shape=(1,),
        order=(0,)
    )
    lse_block_ptr = tl.make_block_ptr(
        lse_ptr,
        shape=(Nt,),
        strides=(stride_lse,),
        offsets=(H_tile_index * H_TILE_SIZE,),
        block_shape=(H_TILE_SIZE,),
        order=(0,)
    )
    targets_tile = tl.load(
        targets_block_ptr,
        boundary_check=(0,),
        padding_option="zero"
    )
    targets_z_tile = tl.zeros((H_TILE_SIZE,), dtype=tl.float32)
    l = tl.zeros((H_TILE_SIZE,), dtype=tl.float32)  # FP32
    m_old = tl.full((H_TILE_SIZE,), -float("inf"), dtype=tl.float32)  # FP32
    for w_start in range(0, V, W_TILE_SIZE):
        z_tile = tl.zeros((H_TILE_SIZE, W_TILE_SIZE), dtype=tl.float32)
        H_block_ptr = tl.make_block_ptr(
            H_ptr,
            shape=(Nt, D),
            strides=(stride_ht, stride_hd),
            offsets=(H_tile_index * H_TILE_SIZE, 0),
            block_shape=(H_TILE_SIZE, D_TILE_SIZE),
            order=(1, 0)
        )
        W_block_ptr = tl.make_block_ptr(
            W_ptr,
            shape=(V, D),
            strides=(stride_wv, stride_wd),
            offsets=(w_start, 0),
            block_shape=(W_TILE_SIZE, D_TILE_SIZE),
            order=(1, 0)
        )
        for _ in range(0, D, D_TILE_SIZE):
            H_tile = tl.load(
                H_block_ptr,
                boundary_check=(0, 1),
                padding_option="zero"
            )
            W_tile = tl.load(
                W_block_ptr,
                boundary_check=(0, 1),
                padding_option="zero"
            )
            z_tile = tl.dot(H_tile, W_tile.T, z_tile, input_precision="ieee")
            H_block_ptr = H_block_ptr.advance((0, D_TILE_SIZE))
            W_block_ptr = W_block_ptr.advance((0, D_TILE_SIZE))
        w_indices = w_start + tl.arange(0, W_TILE_SIZE)
        z_mask = (w_indices < V)[None, :]
        z_tile = tl.where(z_mask, z_tile, -float("inf"))
        z_tile = z_tile.to(H_block_ptr.type.element_ty).to(tl.float32)
        m = tl.maximum(m_old, tl.max(z_tile, axis=-1))
        l = tl.exp(m_old - m) * l + tl.sum(tl.exp(z_tile - m[:, None]), axis=-1)
        targets_z_mask = targets_tile[:, None] == w_indices[None, :]
        targets_z_tile += tl.sum(tl.where(targets_z_mask, z_tile, 0.0), axis=-1)
        m_old = m
    lse_tile = m_old + tl.log(l)
    tl.store(lse_block_ptr, lse_tile, boundary_check=(0,))
    token_mask = (H_tile_index * H_TILE_SIZE + tl.arange(0, H_TILE_SIZE)) < Nt
    token_loss_tile = lse_tile - targets_z_tile
    token_loss_tile = tl.where(token_mask, token_loss_tile, 0.0)
    partial_loss_sum = tl.sum(token_loss_tile)
    tl.store(partial_loss_sums_block_ptr, partial_loss_sum, boundary_check=(0,))

@triton.jit
def fused_lm_head_loss_backward_kernel(
    H_ptr,
    W_ptr,
    targets_ptr,
    lse_ptr,
    dL_ptr,
    dZ_ptr,
    stride_ht, stride_hd,
    stride_wv, stride_wd,
    stride_t,
    stride_lse,
    stride_dzt, stride_dzv,
    H_tile_index,
    Nt,
    D: tl.constexpr,
    V: tl.constexpr,
    H_TILE_SIZE: tl.constexpr,
    W_TILE_SIZE: tl.constexpr,
    D_TILE_SIZE: tl.constexpr,
):
    """Write one [B_t,B_v] tile of the current token tile's G workspace."""

    # Program indices
    W_tile_index = tl.program_id(0)
    dL = tl.load(dL_ptr)
    targets_block_ptr = tl.make_block_ptr(
        targets_ptr,
        shape=(Nt,),
        strides=(stride_t,),
        offsets=(H_tile_index * H_TILE_SIZE,),
        block_shape=(H_TILE_SIZE,),
        order=(0,)
    )
    lse_block_ptr = tl.make_block_ptr(
        lse_ptr,
        shape=(Nt,),
        strides=(stride_lse,),
        offsets=(H_tile_index * H_TILE_SIZE,),
        block_shape=(H_TILE_SIZE,),
        order=(0,)
    )
    targets_tile = tl.load(
        targets_block_ptr,
        boundary_check=(0,),
        padding_option="zero"
    )
    lse_tile = tl.load(
        lse_block_ptr,
        boundary_check=(0,),
        padding_option="zero"
    )
    z_tile = tl.zeros((H_TILE_SIZE, W_TILE_SIZE), dtype=tl.float32)
    H_block_ptr = tl.make_block_ptr(
        H_ptr,
        shape=(Nt, D),
        strides=(stride_ht, stride_hd),
        offsets=(H_tile_index * H_TILE_SIZE, 0),
        block_shape=(H_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0)
    )
    W_block_ptr = tl.make_block_ptr(
        W_ptr,
        shape=(V, D),
        strides=(stride_wv, stride_wd),
        offsets=(W_tile_index * W_TILE_SIZE, 0),
        block_shape=(W_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0)
    )
    for _ in range(0, D, D_TILE_SIZE):
        H_tile = tl.load(
            H_block_ptr,
            boundary_check=(0, 1),
            padding_option="zero"
        )
        W_tile = tl.load(
            W_block_ptr,
            boundary_check=(0, 1),
            padding_option="zero"
        )
        z_tile = tl.dot(H_tile, W_tile.T, z_tile, input_precision="ieee")
        H_block_ptr = H_block_ptr.advance((0, D_TILE_SIZE))
        W_block_ptr = W_block_ptr.advance((0, D_TILE_SIZE))
    w_start = W_tile_index * W_TILE_SIZE
    w_indices = w_start + tl.arange(0, W_TILE_SIZE)
    z_mask = (w_indices < V)[None, :]
    z_tile = tl.where(z_mask, z_tile, -float("inf"))
    z_tile = z_tile.to(H_block_ptr.type.element_ty).to(tl.float32)
    p_tile = tl.exp(z_tile - lse_tile[:, None])
    dZ_tile = dL * (p_tile - (targets_tile[:, None] == w_indices[None, :]))
    dZ_block_ptr = tl.make_block_ptr(
        dZ_ptr,
        shape=(H_TILE_SIZE, V),
        strides=(stride_dzt, stride_dzv),
        offsets=(0, W_TILE_SIZE * W_tile_index),
        block_shape=(H_TILE_SIZE, W_TILE_SIZE),
        order=(1, 0)
    )
    tl.store(dZ_block_ptr, dZ_tile.to(dZ_block_ptr.type.element_ty), boundary_check=(0, 1))


class FusedLMHeadCrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, targets):
        '''
        hidden: (B, S, D)
        weight: (V, D)
        targets: (B, S)
        '''
        assert hidden.ndim == 3
        assert weight.ndim == 2
        assert targets.ndim == 2
        assert hidden.device == weight.device == targets.device
        assert hidden.shape[-1] == weight.shape[-1]
        assert targets.shape == hidden.shape[:-1]
        flat_hidden = hidden.reshape(-1, hidden.shape[-1])
        flat_targets = targets.reshape(-1)
        assert flat_hidden.shape[0] == flat_targets.shape[0]
        
        Nt, D = flat_hidden.shape
        assert D == triton.next_power_of_2(D)
        V = weight.shape[0]
        lse = torch.empty(Nt, dtype=torch.float32, device=hidden.device)
        H_TILE_SIZE = 256
        W_TILE_SIZE = 64
        D_TILE_SIZE = 32
        Kt = math.ceil(Nt / H_TILE_SIZE)
        partial_loss_sums = torch.empty(Kt, dtype=torch.float32, device=hidden.device)
        fused_lm_head_loss_forward_kernel[(Kt,)](
            flat_hidden,
            weight,
            flat_targets,
            partial_loss_sums,
            lse,
            flat_hidden.stride(0), flat_hidden.stride(1),
            weight.stride(0), weight.stride(1),
            flat_targets.stride(0),
            partial_loss_sums.stride(0),
            lse.stride(0),
            Nt,
            Kt,
            D,
            V,
            H_TILE_SIZE,
            W_TILE_SIZE,
            D_TILE_SIZE
        )
        
        ctx.save_for_backward(hidden, weight, targets, lse)
        return partial_loss_sums.sum()

    @staticmethod
    def backward(ctx, grad_out):
        hidden, weight, targets, lse = ctx.saved_tensors
        flat_hidden = hidden.reshape(-1, hidden.shape[-1])
        flat_targets = targets.reshape(-1)
        Nt, D = flat_hidden.shape
        V = weight.shape[0]
        H_TILE_SIZE = 32
        W_TILE_SIZE = 32
        D_TILE_SIZE = 32
        grad_hidden = torch.empty_like(flat_hidden)
        grad_weight = torch.zeros_like(weight, dtype=torch.float32)
        grad_z = torch.empty(H_TILE_SIZE, V, device=hidden.device, dtype=hidden.dtype)
        Kt = math.ceil(Nt / H_TILE_SIZE)
        Kv = math.ceil(V / W_TILE_SIZE)
        for H_tile_index in range(Kt):
            fused_lm_head_loss_backward_kernel[(Kv,)](
                flat_hidden,
                weight,
                flat_targets,
                lse,
                grad_out,
                grad_z,
                flat_hidden.stride(0), flat_hidden.stride(1),
                weight.stride(0), weight.stride(1),
                flat_targets.stride(0),
                lse.stride(0),
                grad_z.stride(0), grad_z.stride(1),
                H_tile_index,
                Nt,
                D,
                V,
                H_TILE_SIZE,
                W_TILE_SIZE,
                D_TILE_SIZE
            )
            h_start = H_tile_index * H_TILE_SIZE
            h_end = min(h_start + H_TILE_SIZE, Nt)
            grad_z_tile = grad_z[:(h_end - h_start), :]
            h_tile = flat_hidden[h_start:h_end, :]
            grad_hidden[h_start:h_end, :] = torch.matmul(grad_z_tile, weight)
            grad_weight += torch.matmul(grad_z_tile.T.to(torch.float32), h_tile.to(torch.float32))
        return grad_hidden.reshape(hidden.shape), grad_weight.to(weight.dtype), None
