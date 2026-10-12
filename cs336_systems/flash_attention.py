import torch
import torch.cuda.nvtx as nvtx
import triton
import triton.language as tl
import math

class FlashAttentionTorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        '''
        Args:
            Q: (B, Nq, d)
            K, V: (B, Nk, d)
        Returns the output.
        '''
        output, L = flash_forward_torch(Q, K, V, is_causal)
        ctx.save_for_backward(Q, K, V, output, L)
        ctx.is_causal = is_causal
        return output

    @staticmethod
    def backward(ctx, grad_out):
        Q, K, V, output, L = ctx.saved_tensors
        is_causal = ctx.is_causal
        return flash_backward_torch(Q, K, V, output, grad_out, L, is_causal)

@torch.compile
def flash_forward_torch(Q, K, V, is_causal=False):
    '''
    Args:
        Q: (B, Nq, d)
        K, V: (B, Nk, d)
    '''
    assert Q.ndim == K.ndim == V.ndim == 3
    assert Q.shape[0] == K.shape[0] == V.shape[0]
    assert Q.shape[-1] == K.shape[-1] == V.shape[-1]
    assert K.shape[-2] == V.shape[-2]
    assert Q.device == K.device == V.device
    assert Q.dtype == K.dtype == V.dtype
    batch_size, Nq, d = Q.shape
    scale = 1.0 / math.sqrt(d)
    Nk = K.shape[-2]
    Bq = 32
    Bk = 32
    Tq = math.ceil(Nq / Bq)
    Tk = math.ceil(Nk / Bk)
    output = torch.empty_like(Q)
    L = torch.empty((batch_size, Nq), dtype=torch.float32, device=Q.device)  # (B, Nq)
    for i in range(Tq):
        q_start = i * Bq
        q_end = min(q_start + Bq, Nq)
        q_tile_size = q_end - q_start
        query_indices = torch.arange(
            q_start,
            q_end,
            device=Q.device,
        )
        Q_tile = Q[:, q_start:q_end, :]  # (B, Bq, d)
        output_tile = torch.zeros_like(Q_tile, dtype=torch.float32)  # (B, Bq, d)
        l = torch.zeros(batch_size, q_tile_size, device=Q.device, dtype=torch.float32)  # (B, Bq)
        m_old = torch.full(
            (batch_size, q_tile_size), 
            -1e6, 
            device=Q.device, 
            dtype=torch.float32
        )  # (B, Bq)
        for j in range(Tk):
            k_start = j * Bk
            k_end = min(k_start + Bk, Nk)
            K_tile = K[:, k_start:k_end, :]  # (B, Bk, d)
            V_tile = V[:, k_start:k_end, :]  # (B, Bk, d)
            S_tile = torch.matmul(
                Q_tile, torch.transpose(K_tile, -2, -1)
            ) * scale  # (B, Bq, Bk)

            if is_causal:
                key_indices = torch.arange(
                    k_start,
                    k_end,
                    device=Q.device,
                )
                causal_mask = (
                    query_indices[:, None]
                    >= key_indices[None, :]
                )

                S_tile = S_tile.masked_fill(
                    ~causal_mask,
                    -1e6,
                )

            m = torch.maximum(m_old, torch.amax(S_tile, dim=-1))
            P_tile = torch.exp(S_tile - m[..., None])  # (B, Bq, Bk)
            alpha = torch.exp(m_old - m)  # (B, Bq)
            l = alpha * l + P_tile.sum(dim=-1)  # (B, Bq)
            output_tile = alpha[..., None] * output_tile + torch.matmul(P_tile.to(V_tile.dtype), V_tile)  # (B, Bq, d)
            m_old = m
        output_tile = output_tile / l[..., None]
        L_tile = m_old + torch.log(l)
        output[:, q_start:q_end, :] = output_tile
        L[:, q_start:q_end] = L_tile
    return output, L

@torch.compile
def flash_backward_torch(Q, K, V, output, grad_out, L, is_causal):
    
    dQ = torch.zeros_like(Q)
    dK = torch.zeros_like(K)
    dV = torch.zeros_like(V)
    D = torch.sum(output.float() * grad_out.float(), dim=-1, dtype=torch.float32)
    _, Nq, d = Q.shape
    scale = 1.0 / math.sqrt(d)
    Nk = K.shape[-2]
    Bq = 32
    Bk = 32
    Tq = math.ceil(Nq / Bq)
    Tk = math.ceil(Nk / Bk)
    for j in range(Tk):
        k_start = j * Bk
        k_end = min(k_start + Bk, Nk)
        key_indices = torch.arange(k_start, k_end, device=K.device)
        K_tile = K[:, k_start:k_end, :]  # (B, Bk, d)
        V_tile = V[:, k_start:k_end, :]  # (B, Bk, d)
        dK_tile = torch.zeros_like(K_tile, dtype=torch.float32)
        dV_tile = torch.zeros_like(V_tile, dtype=torch.float32)
        for i in range(Tq):
            q_start = i * Bq
            q_end = min(q_start + Bq, Nq)
            Q_tile = Q[:, q_start:q_end, :]
            grad_out_tile = grad_out[:, q_start:q_end, :]
            L_tile = L[:, q_start:q_end]
            D_tile = D[:, q_start:q_end]
            S_tile = scale * torch.matmul(
                Q_tile, 
                torch.transpose(K_tile, -2, -1)
            )
            if is_causal:
                query_indices = torch.arange(
                    q_start, q_end, device=Q.device
                )
                causal_mask = (
                    query_indices[:, None]
                    >= key_indices[None, :]
                )
                S_tile = S_tile.masked_fill(
                    ~causal_mask,
                    -1e6,
                )
            P_tile = torch.exp(S_tile - torch.unsqueeze(L_tile, -1))
            dV_tile = dV_tile + torch.matmul(
                torch.transpose(P_tile, -2, -1).to(grad_out_tile.dtype), grad_out_tile
            )
            dP_tile = torch.matmul(
                grad_out_tile, 
                torch.transpose(V_tile, -2, -1)
            )
            dS_tile = P_tile * (dP_tile - torch.unsqueeze(D_tile, -1))
            dK_tile = dK_tile + scale * torch.matmul(
                torch.transpose(dS_tile, -2, -1).to(Q_tile.dtype), Q_tile
            )
        dK[:, k_start:k_end, :] = dK_tile
        dV[:, k_start:k_end, :] = dV_tile
    for i in range(Tq):
        q_start = i * Bq
        q_end = min(q_start + Bq, Nq)
        query_indices = torch.arange(
            q_start, q_end, device=Q.device
        )
        Q_tile = Q[:, q_start:q_end, :]
        grad_out_tile = grad_out[:, q_start:q_end, :]
        dQ_tile = torch.zeros_like(Q_tile, dtype=torch.float32)
        L_tile = L[:, q_start:q_end]
        D_tile = D[:, q_start:q_end]
        for j in range(Tk):
            k_start = j * Bk
            k_end = min(k_start + Bk, Nk)
            K_tile = K[:, k_start:k_end, :]
            V_tile = V[:, k_start:k_end, :]
            S_tile = scale * torch.matmul(
                Q_tile, 
                torch.transpose(K_tile, -2, -1)
            )
            if is_causal:
                key_indices = torch.arange(
                    k_start, k_end, device=K.device
                )
                causal_mask = (
                    query_indices[:, None]
                    >= key_indices[None, :]
                )
                S_tile = S_tile.masked_fill(
                    ~causal_mask,
                    -1e6,
                )
            P_tile = torch.exp(S_tile - torch.unsqueeze(L_tile, -1))
            dP_tile = torch.matmul(
                grad_out_tile, 
                torch.transpose(V_tile, -2, -1)
            )
            dS_tile = P_tile * (dP_tile - torch.unsqueeze(D_tile, -1))
            dQ_tile = dQ_tile + torch.matmul(dS_tile.to(K_tile.dtype), K_tile) * scale
        dQ[:, q_start:q_end, :] = dQ_tile
    return dQ, dK, dV, None


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
        #S_tile = tl.dot(Q_tile, K_tile.T, input_precision="ieee") * scale  # for correctness check
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
        #output = alpha[:, None] * output + tl.dot(P_tile.to(V_tile.dtype), V_tile, input_precision="ieee")  # for correctness check
        m_old = m
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))
        
    output = output / l[:, None]
    L = m_old + tl.log(l)  # FP32
    tl.store(O_block_ptr, output.to(O_block_ptr.type.element_ty), boundary_check=(0, 1))
    tl.store(L_block_ptr, L, boundary_check=(0,))

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
        #S_tile = tl.dot(Q_tile, K_tile.T, input_precision="ieee") * scale  # for correctness check
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
        #dP_tile = tl.dot(dO_tile, V_tile.T, input_precision="ieee")  # for correctness check
        dS_tile = P_tile * (dP_tile - D_tile[:, None])  # FP32
        dQ_tile = dQ_tile + tl.dot(dS_tile.to(K_tile.dtype), K_tile) * scale
        #dQ_tile = dQ_tile + tl.dot(dS_tile.to(K_tile.dtype), K_tile, input_precision="ieee") * scale  # for correctness check
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))
    tl.store(dQ_block_ptr, dQ_tile.to(dQ_block_ptr.type.element_ty), boundary_check=(0, 1))

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
        #S_tile = tl.dot(Q_tile, K_tile.T, input_precision="ieee") * scale  # for correctness check
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
        #dV_tile = dV_tile + tl.dot(P_tile.to(dO_tile.dtype).T, dO_tile, input_precision="ieee")  # for correctness check
        dP_tile = tl.dot(dO_tile, V_tile.T)  # FP32
        #dP_tile = tl.dot(dO_tile, V_tile.T, input_precision="ieee")  # for correctness check
        dS_tile = P_tile * (dP_tile - D_tile[:, None])
        dK_tile = dK_tile + tl.dot(dS_tile.to(Q_tile.dtype).T, Q_tile) * scale
        #dK_tile = dK_tile + tl.dot(dS_tile.to(Q_tile.dtype).T, Q_tile, input_precision="ieee") * scale  # for correctness check
        Q_block_ptr = Q_block_ptr.advance((Q_TILE_SIZE, 0))
        dO_block_ptr = dO_block_ptr.advance((Q_TILE_SIZE, 0))
        L_block_ptr = L_block_ptr.advance((Q_TILE_SIZE,))
        D_block_ptr = D_block_ptr.advance((Q_TILE_SIZE,))
    tl.store(dK_block_ptr, dK_tile.to(dK_block_ptr.type.element_ty), boundary_check=(0, 1))
    tl.store(dV_block_ptr, dV_tile.to(dV_block_ptr.type.element_ty), boundary_check=(0, 1))

class FlashAttentionTriton(torch.autograd.Function):
    @staticmethod
    @nvtx.range("flash attention forward")
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
        # A100 80GB
        # (32 heads, 32,768 tokens, head dimension 128, BF16, causal).
        Q_TILE_SIZE = 128
        K_TILE_SIZE = 32
        Tq = math.ceil(n_queries / Q_TILE_SIZE)
        L = torch.empty(
            (batch_size, n_queries), 
            dtype=torch.float32, 
            device=Q.device
        )  # FP32
        scale = 1 / math.sqrt(d)
        _flash_fwd[(Tq, batch_size)](
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
            Q_TILE_SIZE=Q_TILE_SIZE,
            K_TILE_SIZE=K_TILE_SIZE,
            is_causal=is_causal,
            num_warps=4,
            num_stages=2,
        )
        ctx.save_for_backward(Q, K, V, output, L) 
        ctx.is_causal = is_causal
        return output

    @staticmethod
    @nvtx.range("flash attention backward")
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
        DQ_Q_TILE_SIZE = 64
        DQ_K_TILE_SIZE = 64
        DKDV_Q_TILE_SIZE = 32
        DKDV_K_TILE_SIZE = 64
        Tq = math.ceil(n_queries / DQ_Q_TILE_SIZE)
        scale = 1 / math.sqrt(d)
        _flash_bwd_dq[(Tq, batch_size)](
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
            Q_TILE_SIZE=DQ_Q_TILE_SIZE,
            K_TILE_SIZE=DQ_K_TILE_SIZE,
            is_causal=is_causal,
            num_warps=4,
            num_stages=2,
        )

        Tk = math.ceil(n_keys / DKDV_K_TILE_SIZE)
        _flash_bwd_dkdv[(Tk, batch_size)](
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
            Q_TILE_SIZE=DKDV_Q_TILE_SIZE,
            K_TILE_SIZE=DKDV_K_TILE_SIZE,
            is_causal=is_causal,
            num_warps=4,
            num_stages=3,
        )
        
        return dQ, dK, dV, None
     