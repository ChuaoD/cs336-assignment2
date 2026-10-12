import itertools
import math
import torch
import triton
import triton.language as tl
from pathlib import Path

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def make_configs(tile_sizes, num_warps=(4, 8), num_stages=(2, 3)):
    """Every combination of (Q_TILE_SIZE, K_TILE_SIZE), num_warps, and num_stages.

    Configs that need more shared memory than the GPU has fail to compile;
    the autotuner records them as infinitely slow and moves on.
    """
    return [
        triton.Config(
            {"Q_TILE_SIZE": q_tile, "K_TILE_SIZE": k_tile},
            num_warps=warps,
            num_stages=stages,
        )
        for (q_tile, k_tile), warps, stages in itertools.product(
            tile_sizes, num_warps, num_stages
        )
    ]


# Each search includes (32, 64) with four warps and two stages, the config
# flash_attention.py uses today, as the baseline. The forward and dQ kernels
# keep a Q tile resident and stream K/V tiles past it, so they favor large Q
# tiles; the dK/dV kernel keeps a K/V tile resident and streams Q tiles.
FWD_CONFIGS = make_configs(
    [(32, 64), (64, 32), (64, 64), (64, 128), (128, 32), (128, 64), (128, 128)],
    num_stages=(2, 3, 4),
)
DQ_CONFIGS = make_configs([(32, 64), (64, 32), (64, 64), (128, 32), (128, 64)])
DKDV_CONFIGS = make_configs([(32, 64), (64, 64), (32, 128), (64, 128)])
AUTOTUNE_KEY = ["N_QUERIES", "N_KEYS", "D", "is_causal"]

@triton.autotune(configs=FWD_CONFIGS, key=AUTOTUNE_KEY)
@triton.jit
def _flash_fwd(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr
):
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0)
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    Q_tile = tl.load(
        Q_block_ptr, 
        boundary_check=(0, 1), 
        padding_option="zero"
    )
    output = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)  # FP32
    l = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)  # FP32
    m_old = tl.full((Q_TILE_SIZE,), -float("inf"), dtype=tl.float32)  # FP32
    query_start = query_tile_index * Q_TILE_SIZE
    query_end = query_start + Q_TILE_SIZE
    query_indices = query_start + tl.arange(0, Q_TILE_SIZE)
    query_limit = tl.minimum(N_QUERIES, query_end)
    if is_causal:
        key_limit = tl.minimum(N_KEYS, query_limit)
    else:
        key_limit = N_KEYS
    for key_start in range(0, key_limit, K_TILE_SIZE):
        key_indices = key_start + tl.arange(0, K_TILE_SIZE)
        K_tile = tl.load(
            K_block_ptr, 
            boundary_check=(0, 1), 
            padding_option="zero"
        )
        V_tile = tl.load(
            V_block_ptr, 
            boundary_check=(0, 1), 
            padding_option="zero"
        )
        S_tile = tl.dot(Q_tile, K_tile.T) * scale  # FP32
        score_mask = (
            (query_indices[:, None] < N_QUERIES)
            &
            (key_indices[None, :] < N_KEYS)
        )
        if is_causal:
            score_mask = (
                score_mask
                & (
                    query_indices[:, None] 
                    >= key_indices[None, :]
                )    
            )
        S_tile = tl.where(score_mask, S_tile, -1e6)
        m = tl.maximum(m_old, tl.max(S_tile, axis=-1))
        P_tile = tl.exp(S_tile - m[:, None])  # FP32
        alpha = tl.exp(m_old - m)
        l = alpha * l + tl.sum(P_tile, axis=-1)
        output = alpha[:, None] * output + tl.dot(P_tile.to(V_tile.dtype), V_tile)
        m_old = m
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))
        
    output = output / l[:, None]
    L = m_old + tl.log(l)  # FP32
    tl.store(O_block_ptr, output.to(O_block_ptr.type.element_ty), boundary_check=(0, 1))
    tl.store(L_block_ptr, L, boundary_check=(0,))

@triton.autotune(configs=DQ_CONFIGS, key=AUTOTUNE_KEY)
@triton.jit
def _flash_bwd_dq(
    Q_ptr, K_ptr, V_ptr,
    dO_ptr, L_ptr, D_ptr,
    dQ_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_dob, stride_doq, stride_dod,
    stride_lb, stride_lq,
    stride_db, stride_dq,
    stride_dqb, stride_dqq, stride_dqd,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr
):
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    
    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_index * stride_dob,
        shape=(N_QUERIES, D),
        strides=(stride_doq, stride_dod),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0)
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )
    D_block_ptr = tl.make_block_ptr(
        D_ptr + batch_index * stride_db,
        shape=(N_QUERIES,),
        strides=(stride_dq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )
    dQ_block_ptr = tl.make_block_ptr(
        dQ_ptr + batch_index * stride_dqb,
        shape=(N_QUERIES, D),
        strides=(stride_dqq, stride_dqd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    Q_tile = tl.load(
        Q_block_ptr, 
        boundary_check=(0, 1), 
        padding_option="zero"
    )
    dO_tile = tl.load(dO_block_ptr,
        boundary_check=(0, 1), 
        padding_option="zero"
    )
    L_tile = tl.load(L_block_ptr,
        boundary_check=(0,),
        padding_option="zero"
    )
    D_tile = tl.load(D_block_ptr,
        boundary_check=(0,),
        padding_option="zero"
    )
    dQ_tile = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)  # FP32
    query_start = query_tile_index * Q_TILE_SIZE
    query_end = query_start + Q_TILE_SIZE
    query_indices = query_start + tl.arange(0, Q_TILE_SIZE)
    query_limit = tl.minimum(N_QUERIES, query_end)
    if is_causal:
        key_limit = tl.minimum(N_KEYS, query_limit)
    else:
        key_limit = N_KEYS
    for key_start in range(0, key_limit, K_TILE_SIZE):
        key_indices = key_start + tl.arange(0, K_TILE_SIZE)
        K_tile = tl.load(
            K_block_ptr, 
            boundary_check=(0, 1), 
            padding_option="zero"
        )
        V_tile = tl.load(
            V_block_ptr, 
            boundary_check=(0, 1), 
            padding_option="zero"
        )
        S_tile = tl.dot(Q_tile, K_tile.T) * scale  # FP32
        score_mask = (
            (query_indices[:, None] < N_QUERIES)
            &
            (key_indices[None, :] < N_KEYS)
        )
        if is_causal:
            score_mask = (
                score_mask
                & (
                    query_indices[:, None] 
                    >= key_indices[None, :]
                )    
            )
        S_tile = tl.where(score_mask, S_tile, -1e6)
        P_tile = tl.exp(S_tile - L_tile[:, None])  # FP32
        dP_tile = tl.dot(dO_tile, V_tile.T)  # FP32
        dS_tile = P_tile * (dP_tile - D_tile[:, None])  # FP32
        dQ_tile = dQ_tile + tl.dot(dS_tile.to(K_tile.dtype), K_tile) * scale
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))
    tl.store(dQ_block_ptr, dQ_tile.to(dQ_block_ptr.type.element_ty), boundary_check=(0, 1))

@triton.autotune(configs=DKDV_CONFIGS, key=AUTOTUNE_KEY)
@triton.jit
def _flash_bwd_dkdv(
    Q_ptr, K_ptr, V_ptr,
    dO_ptr, L_ptr, D_ptr,
    dK_ptr, dV_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_dob, stride_doq, stride_dod,
    stride_lb, stride_lq,
    stride_db, stride_dq,
    stride_dkb, stride_dkk, stride_dkd,
    stride_dvb, stride_dvk, stride_dvd,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr
):
    # Program indices
    key_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    
    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    dK_block_ptr = tl.make_block_ptr(
        dK_ptr + batch_index * stride_dkb,
        shape=(N_KEYS, D),
        strides=(stride_dkk, stride_dkd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    dV_block_ptr = tl.make_block_ptr(
        dV_ptr + batch_index * stride_dvb,
        shape=(N_KEYS, D),
        strides=(stride_dvk, stride_dvd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    K_tile = tl.load(
        K_block_ptr, 
        boundary_check=(0, 1), 
        padding_option="zero"
    )
    V_tile = tl.load(
        V_block_ptr, 
        boundary_check=(0, 1), 
        padding_option="zero"
    )
    dK_tile = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)  # FP32
    dV_tile = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)  # FP32
    key_start = key_tile_index * K_TILE_SIZE
    key_indices = key_start + tl.arange(0, K_TILE_SIZE)
    if is_causal:
        query_limit = key_start
    else:
        query_limit = 0
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_limit, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_index * stride_dob,
        shape=(N_QUERIES, D),
        strides=(stride_doq, stride_dod),
        offsets=(query_limit, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0)
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_limit,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )
    D_block_ptr = tl.make_block_ptr(
        D_ptr + batch_index * stride_db,
        shape=(N_QUERIES,),
        strides=(stride_dq,),
        offsets=(query_limit,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    for query_start in range(query_limit, N_QUERIES, Q_TILE_SIZE):
        query_indices = query_start + tl.arange(0, Q_TILE_SIZE)
        Q_tile = tl.load(
            Q_block_ptr, 
            boundary_check=(0, 1), 
            padding_option="zero"
        )
        dO_tile = tl.load(dO_block_ptr,
            boundary_check=(0, 1), 
            padding_option="zero"
        )
        L_tile = tl.load(L_block_ptr,
            boundary_check=(0,),
            padding_option="zero"
        )
        D_tile = tl.load(D_block_ptr,
            boundary_check=(0,),
            padding_option="zero"
        )
        S_tile = tl.dot(Q_tile, K_tile.T) * scale  # FP32
        score_mask = (
            (query_indices[:, None] < N_QUERIES)
            &
            (key_indices[None, :] < N_KEYS)
        )

        if is_causal:
            score_mask = (
                score_mask
                & (
                    query_indices[:, None] 
                    >= key_indices[None, :]
                )    
            )
        S_tile = tl.where(score_mask, S_tile, -1e6)
        P_tile = tl.exp(S_tile - L_tile[:, None])  # FP32
        dV_tile = dV_tile + tl.dot(P_tile.to(dO_tile.dtype).T, dO_tile)  # FP32
        dP_tile = tl.dot(dO_tile, V_tile.T)  # FP32
        dS_tile = P_tile * (dP_tile - D_tile[:, None])
        dK_tile = dK_tile + tl.dot(dS_tile.to(Q_tile.dtype).T, Q_tile) * scale
        Q_block_ptr = Q_block_ptr.advance((Q_TILE_SIZE, 0))
        dO_block_ptr = dO_block_ptr.advance((Q_TILE_SIZE, 0))
        L_block_ptr = L_block_ptr.advance((Q_TILE_SIZE,))
        D_block_ptr = D_block_ptr.advance((Q_TILE_SIZE,))
    tl.store(dK_block_ptr, dK_tile.to(dK_block_ptr.type.element_ty), boundary_check=(0, 1))
    tl.store(dV_block_ptr, dV_tile.to(dV_block_ptr.type.element_ty), boundary_check=(0, 1))

class FlashAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        assert Q.ndim == K.ndim == V.ndim == 3
        assert Q.shape[0] == K.shape[0] == V.shape[0]
        assert Q.shape[-1] == K.shape[-1] == V.shape[-1]
        assert K.shape[-2] == V.shape[-2]
        assert Q.device == K.device == V.device
        assert Q.dtype == K.dtype == V.dtype

        output = torch.empty_like(Q)  # Q, K, V, O have the same dtype
        batch_size, n_queries, d = Q.shape
        assert d == triton.next_power_of_2(d)
        n_keys = K.shape[-2]
        #Q_TILE_SIZE = 32
        #K_TILE_SIZE = 32
        #Tq = math.ceil(n_queries / Q_TILE_SIZE)
        L = torch.empty(
            (batch_size, n_queries), 
            dtype=torch.float32, 
            device=Q.device
        )  # FP32
        scale = 1 / math.sqrt(d)
        grid = lambda meta: (
            triton.cdiv(n_queries, meta["Q_TILE_SIZE"]), 
            batch_size
        )
        _flash_fwd[grid](
            Q, K, V,
            output, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            output.stride(0), output.stride(1), output.stride(2),
            L.stride(0), L.stride(1),
            N_QUERIES=n_queries, N_KEYS=n_keys,
            scale=scale,
            D=d,
            is_causal=is_causal
        )
        ctx.save_for_backward(Q, K, V, output, L) 
        ctx.is_causal = is_causal
        return output

    @staticmethod
    def backward(ctx, grad_out):
        Q, K, V, output, L = ctx.saved_tensors
        is_causal = ctx.is_causal
        D = torch.sum(output.float() * grad_out.float(), dim=-1, dtype=torch.float32)  # FP32
        dQ = torch.empty_like(Q)  # same dtype as Q
        dK = torch.empty_like(K)
        dV = torch.empty_like(V)
        batch_size, n_queries, d = Q.shape
        assert d == triton.next_power_of_2(d)
        n_keys = K.shape[-2]
        #Q_TILE_SIZE = 32
        #K_TILE_SIZE = 32
        #Tq = math.ceil(n_queries / Q_TILE_SIZE)
        scale = 1 / math.sqrt(d)
        grid = lambda meta: (
            triton.cdiv(n_queries, meta["Q_TILE_SIZE"]), 
            batch_size
        )
        _flash_bwd_dq[grid](
            Q, K, V,
            grad_out, L, D,
            dQ,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            grad_out.stride(0,), grad_out.stride(1), grad_out.stride(2),
            L.stride(0), L.stride(1),
            D.stride(0), D.stride(1),
            dQ.stride(0), dQ.stride(1), dQ.stride(2),
            N_QUERIES=n_queries, N_KEYS=n_keys,
            scale=scale,
            D=d,
            is_causal=is_causal
        )

        # Tk = math.ceil(n_keys / K_TILE_SIZE)
        grid = lambda meta: (
            triton.cdiv(n_keys, meta["K_TILE_SIZE"]), 
            batch_size
        )
        _flash_bwd_dkdv[grid](
            Q, K, V,
            grad_out, L, D,
            dK, dV, 
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            grad_out.stride(0,), grad_out.stride(1), grad_out.stride(2),
            L.stride(0), L.stride(1),
            D.stride(0), D.stride(1),
            dK.stride(0), dK.stride(1), dK.stride(2),
            dV.stride(0), dV.stride(1), dV.stride(2),
            N_QUERIES=n_queries, N_KEYS=n_keys,
            scale=scale,
            D=d,
            is_causal=is_causal
        )
        
        return dQ, dK, dV, None
    
flash_triton = FlashAttentionTriton.apply

BATCH = 32
is_causal = True
D = 128
SEQ_LEN = 32768
dtype = torch.bfloat16

                     
def print_autotuning_results(kernel, name, flops):
    """Print every config's median time, fastest first."""
    timings = getattr(kernel, "configs_timings", None)
    if timings is None:
        # Older Triton releases keep only the winner.
        print(f"== {name}: best config {kernel.best_config} ==")
        return
    rows = []
    for config, timing in timings.items():
        # The autotuner stores [median, 20th, 80th percentile] in milliseconds.
        ms = timing[0] if isinstance(timing, (list, tuple)) else timing
        rows.append((ms, config))
    print(f"== {name}: {len(rows)} configs ==")
    for ms, config in sorted(rows, key=lambda row: row[0]):
        tiles = f"Q={config.kwargs['Q_TILE_SIZE']:>3} K={config.kwargs['K_TILE_SIZE']:>3}"
        launch = f"warps={config.num_warps} stages={config.num_stages}"
        if math.isinf(ms):
            print(f"  {tiles}  {launch}  skipped (does not fit)")
        else:
            tflops = flops / (ms / 1_000) / 1e12
            print(f"  {tiles}  {launch}  {ms:9.3f} ms  {tflops:6.1f} TFLOP/s")


def main():
    
    Q = torch.randn(
        BATCH, SEQ_LEN, D,
        device=DEVICE,
        dtype=dtype,
        requires_grad=True,
    )
    K = torch.randn_like(Q, requires_grad=True)
    V = torch.randn_like(Q, requires_grad=True)

    output = FlashAttentionTriton.apply(Q, K, V, is_causal)
    grad_output = torch.randn_like(output)
    output.backward(grad_output)
    torch.cuda.synchronize()

    # One causal [N, N] x [N, D] product per batch entry; the mask removes half.
    causal_matmul_flops = BATCH * SEQ_LEN**2 * D
    # Forward: QK^T, PV. dQ: QK^T, dO V^T, dS K. dK/dV: QK^T, P^T dO, dO V^T, dS^T Q.
    print_autotuning_results(_flash_fwd, "forward", 2 * causal_matmul_flops)
    print_autotuning_results(_flash_bwd_dq, "backward dQ", 3 * causal_matmul_flops)
    print_autotuning_results(_flash_bwd_dkdv, "backward dK/dV", 4 * causal_matmul_flops)


if __name__ == "__main__":
    main()
