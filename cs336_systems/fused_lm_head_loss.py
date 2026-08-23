from dataclasses import dataclass
from typing import Literal
from jaxtyping import Bool, Float, Int
import math
import torch
from torch import Tensor
import triton
import triton.language as tl
from einops import einsum, rearrange



def materialized_lm_head_cross_entropy(
    hidden: Float[Tensor, "Nt D"],
    weight: Float[Tensor, "V D"],
    targets: Int[Tensor, "Nt"],
) -> torch.Tensor:
    logits = einsum(hidden, weight, "Nt D, V D -> Nt V")
    m = logits.amax(dim=-1, keepdim=True)
    log_sum_exp = torch.log(torch.exp(logits - m).sum(-1)) + m.squeeze(-1)  # (Nt,)
    loss = log_sum_exp - torch.gather(logits, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    return loss.sum()

def tiled_lm_head_cross_entropy_forward(
    hidden: Float[Tensor, "Nt D"],
    weight: Float[Tensor, "V D"],
    targets: Int[Tensor, "Nt"],
    *,
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
    token_losses_ptr,
    lse_ptr,
    stride_ht, stride_hd,
    stride_wv, stride_wd,
    stride_t,
    stride_tl,
    stride_lse,
    Nt,
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
    token_losses_block_ptr = tl.make_block_ptr(
        token_losses_ptr,
        shape=(Nt,),
        strides=(stride_tl,),
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
    targets_z_tile = tl.zeros((H_TILE_SIZE,), dtype=tl.float32)
    l = tl.zeros((H_TILE_SIZE,), dtype=tl.float32)  # FP32
    m_old = tl.full((H_TILE_SIZE,), -float("inf"), dtype=tl.float32)  # FP32
    for w_start in range(0, V, W_TILE_SIZE):
        w_end = tl.minimum(w_start + W_TILE_SIZE, V)
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
            z_tile = tl.dot(H_tile, W_tile.T, z_tile)
            H_block_ptr = H_block_ptr.advance((0, D_TILE_SIZE))
            W_block_ptr = W_block_ptr.advance((0, D_TILE_SIZE))
        w_indices = w_start + tl.arange(0, W_TILE_SIZE)
        z_mask = tl.expand_dims(w_indices < V, 0)
        z_tile = tl.where(z_mask, z_tile, -float("inf")) 
        m = tl.maximum(m_old, tl.max(z_tile, axis=-1))
        l = tl.exp(m_old - m) * l + tl.sum(
            tl.exp(z_tile - tl.expand_dims(m, 1)),
            axis=-1,
        )
        targets_in_tile = (targets_tile >= w_start) & (targets_tile < w_end)
        target_columns = (
            tl.expand_dims(targets_tile, 1)
            == tl.expand_dims(w_indices, 0)
        )
        candidate_targets_z = tl.sum(
            tl.where(target_columns, z_tile, 0.0),
            axis=-1,
        )
        targets_z_tile = tl.where(
            targets_in_tile,
            candidate_targets_z,
            targets_z_tile
        )
        m_old = m
    lse_tile = m + tl.log(l)
    tl.store(lse_block_ptr, lse_tile, boundary_check=(0,))
    token_loss_tile = lse_tile - targets_z_tile
    tl.store(token_losses_block_ptr, token_loss_tile, boundary_check=(0,))


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
        token_losses = torch.empty(Nt, dtype=torch.float32, device=hidden.device)
        lse = torch.empty(Nt, dtype=torch.float32, device=hidden.device)
        H_TILE_SIZE = 32
        W_TILE_SIZE = 32
        D_TILE_SIZE = 32
        Kt = math.ceil(Nt / H_TILE_SIZE)
        fused_lm_head_loss_forward_kernel[(Kt,)](
            flat_hidden,
            weight,
            flat_targets,
            token_losses,
            lse,
            flat_hidden.stride(0), flat_hidden.stride(1),
            weight.stride(0), weight.stride(1),
            flat_targets.stride(0),
            token_losses.stride(0),
            lse.stride(0),
            Nt,
            D,
            V,
            H_TILE_SIZE,
            W_TILE_SIZE,
            D_TILE_SIZE
        )
        
        ctx.save_for_backward(hidden, weight, targets, lse)
        return token_losses.sum()
