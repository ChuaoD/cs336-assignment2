from __future__ import annotations

from cs336_basics.model import (
    Linear, 
    Embedding, 
    RotaryPositionalEmbedding, 
    SwiGLU,
    RMSNorm
)

from cs336_systems.flash_attention import FlashAttentionTriton
from cs336_systems.fused_lm_head_loss import FusedLMHeadCrossEntropyFunction

import torch
import torch.nn as nn
from einops import rearrange, einsum

from jaxtyping import Float, Int
from torch import Tensor


class FlashMultiHeadAttention(torch.nn.Module):
    def __init__(self, 
                 d_model: int,
                 num_heads: int,
                 with_rope: bool = False,
                 theta: float = None,
                 max_seq_len: int = None,
                 device: torch.device | None = None
        ) -> None:

        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.with_rope = with_rope
        self.head_dim = d_model // num_heads
        assert self.head_dim > 0
        assert self.head_dim & (self.head_dim - 1) == 0, (
            "The current Triton kernel requires a power-of-two head dimension"
        )
        if self.with_rope:
            self.rope = RotaryPositionalEmbedding(theta, self.head_dim, max_seq_len, device)
        self.q_proj = Linear(d_model, num_heads * self.head_dim, device=device)
        self.k_proj = Linear(d_model, num_heads * self.head_dim, device=device)
        self.v_proj = Linear(d_model, num_heads * self.head_dim, device=device)
        self.output_proj = Linear(num_heads * self.head_dim, d_model, device=device)

    def forward(self, 
                x: Float[Tensor, "batch_size seq_len d_model"],
                token_positions: Int[Tensor, "batch_size num_heads seq_len"] | None = None
    ):
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        Q, K, V = (
            rearrange(M, "B N (H Dh) -> B H N Dh", H=self.num_heads)
            for M in (Q, K, V)
        )

        # Apply RoPE
        if self.with_rope:
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        Q, K, V = (
            rearrange(M, "B H N Dh -> (B H) N Dh") for M in (Q, K, V)
        )

        attn = FlashAttentionTriton.apply(Q, K, V, True)
        attn = rearrange(attn, "(B H) N Dh -> B N (H Dh)", H=self.num_heads)

        output = self.output_proj(attn)

        return output

class FlashTransformerBlock(nn.Module):
    def __init__(self,
                 d_model: int,
                 num_heads: int, 
                 d_ff: int,
                 theta: float | None = None,
                 max_seq_len: int | None = None,
                 device: torch.device | None = None
    ) -> None:
        super().__init__()
        self.ln1 = RMSNorm(d_model, device=device)
        self.ln2 = RMSNorm(d_model, device=device)
        self.ffn = SwiGLU(d_model, d_ff, device)
        self.attn = FlashMultiHeadAttention(d_model, num_heads, 
                                            with_rope=True,
                                            theta=theta,
                                            max_seq_len=max_seq_len, device=device)
    
    def forward(self, x: Tensor, token_positions: Tensor | None = None):
        x_attn = self.attn(self.ln1(x), token_positions)
        out1 = x + x_attn
        return out1 + self.ffn(self.ln2(out1))
    
class TransformerLM(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 context_length: int,
                 num_layers: int,
                 d_model: int,
                 num_heads: int,
                 d_ff: int,
                 theta: float | None = None,
                 device: torch.device | None = None
    ) -> None:
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model, device=device)
        self.layers = nn.ModuleList([
            FlashTransformerBlock(
                d_model, num_heads, d_ff, theta=theta, max_seq_len=context_length, device=device
            ) 
            for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model, device=device)
        self.lm_head = LMHead(d_model, vocab_size, device=device)
        self.context_length = context_length

    def forward(self, x: Tensor, token_positions: Tensor | None = None, targets: Tensor | None = None):
        out = self.token_embeddings(x)
        for layer in self.layers:
            out = layer(out, token_positions)
        out= self.ln_final(out)
        return self.lm_head(out, targets)

class LMHead(Linear):
    def __init__(self, 
                 in_features: int, 
                 out_features: int, 
                 device: torch.device | None = None, 
                 dtype: torch.dtype | None = None
    ) -> None:
        '''
        Construct a linear transformation module. 
        Args:
            in_features: final dimension of the input
            out_features: final dimension of the output
            device: Device to store the parameters on
            dtype: Data type of the parameters
        '''
        super().__init__(in_features, out_features, device=device, dtype=dtype)
        
    def forward(self, hidden, targets=None) -> Tensor:
        if targets is not None:
            return FusedLMHeadCrossEntropyFunction.apply(hidden, self.weight, targets)
        return einsum(hidden, self.weight, "... d_in, d_out d_in -> ... d_out")
        