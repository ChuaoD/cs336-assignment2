from __future__ import annotations

from cs336_basics.model import (
    Linear, 
    Embedding, 
    RotaryPositionalEmbedding, 
    SwiGLU,
    RMSNorm
)
from cs336_basics.optimizer import AdamW
from cs336_basics.utils import (
    cross_entropy_loss,
    softmax
)
from cs336_systems.flash_attention import FlashAttentionTriton

import torch
import torch.nn as nn
from einops import rearrange

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
        self.lm_head = Linear(d_model, vocab_size, device=device)
        self.context_length = context_length

    def forward(self, x: Tensor, token_positions: Tensor | None = None):
        out = self.token_embeddings(x)
        for layer in self.layers:
            out = layer(out, token_positions)
        out= self.ln_final(out)
        return self.lm_head(out)

def benchmark(
        mode: str,
        d_model: int,
        d_ff: int,
        num_layers: int,
        num_heads: int,
        batch_size: int = 4,
        context_length: int = 512,
        vocab_size: int = 10000,
        device: str | torch.device = "cuda",
        theta: float = 10000.0,
        warmup_steps: int = 5,
        time_steps: int = 10,
        profile: bool = False,
        compiled: bool = False
):

    device = torch.device(device)
    torch.set_float32_matmul_precision('high')

    if profile:
        basics_model.attention = annotated_attention
    if compiled:
        basics_model.TransformerLM = torch.compile(TransformerLM)

    model = TransformerLM(
        vocab_size,
        context_length,
        num_layers,
        d_model,
        num_heads,
        d_ff,
        theta,
        device
    )

    input_batch = torch.randint(
        0, vocab_size, 
        (batch_size, context_length), 
        dtype=torch.long,
        device=device
    )

    target_batch = torch.randint(
        0, vocab_size, 
        (batch_size, context_length), 
        dtype=torch.long,
        device=device
    )

    optimizer_config = {
        "lr": 1e-3,
        "betas": (0.9, 0.95),
        "eps": 1e-6,
        "weight_decay": 0.01
    }

    optimizer = AdamW(model.parameters(), **optimizer_config)

    model.train()

    statistics = {}

    if profile:
        # Warm up outside the captured NVTX range.
        for _ in range(warmup_steps):
            optimizer.zero_grad(set_to_none=True)

            logits = model(input_batch)

            if mode != "forward":
                loss = cross_entropy_loss(logits, target_batch)
                loss.backward()

            if mode == "full":
                optimizer.step()

            torch.cuda.synchronize(device)

        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device)

        # Nsight starts collecting when this range opens.
        with nvtx.range("profile"):
            with nvtx.range("forward"):
                logits = model(input_batch)
                torch.cuda.synchronize(device)

            if mode != "forward":
                with nvtx.range("loss"):
                    loss = cross_entropy_loss(logits, target_batch)
                    torch.cuda.synchronize(device)

                with nvtx.range("backward"):
                    loss.backward()
                    torch.cuda.synchronize(device)

            if mode == "full":
                with nvtx.range("optimizer"):
                    optimizer.step()
                    torch.cuda.synchronize(device)

        return {"profiled_steps": 1}

    # Warm up
    for _ in range(warmup_steps):
        optimizer.zero_grad(set_to_none=True)

        logits = model(input_batch)

        if mode != "forward":
            loss = cross_entropy_loss(logits, target_batch)
            loss.backward()

        if mode == "full":
            optimizer.step()

        torch.cuda.synchronize(device)
        del logits

    if mode == "forward":
        # Time the forward step
        forward_times = np.zeros(time_steps)    
        for i in range(time_steps):
            # Clear gradients from the previous step
            optimizer.zero_grad()
            # Forward pass
            torch.cuda.synchronize(device)
            forward_start = default_timer()
            logits = model(input_batch)
            torch.cuda.synchronize(device)
            forward_end = default_timer()
            forward_times[i] = forward_end - forward_start
            del logits
        statistics["forward_mean"] = (forward_times * 1000).mean()
        statistics["forward_std"] = (forward_times * 1000).std()

    elif mode == "forward-backward":
        # Time the forward and backward step
        forward_backward_times = np.zeros(time_steps)
        for i in range(time_steps):
            # Clear gradients from the previous step
            optimizer.zero_grad()
            # Forward pass
            torch.cuda.synchronize()
            forward_start = default_timer()
            logits = model(input_batch)
            # Compute loss
            loss = cross_entropy_loss(logits, target_batch)
            # Compute gradients
            loss.backward()
            torch.cuda.synchronize()
            backward_end = default_timer()
            forward_backward_times[i] = backward_end - forward_start
    
        statistics["forward-backward_mean"] = (forward_backward_times * 1000).mean()
        statistics["forward-backward_std"] = (forward_backward_times * 1000).std()
    elif mode == "full":
        # Time the full training step
        full_times = np.zeros(time_steps)      
        for i in range(time_steps):
            # Clear gradients from the previous step
            optimizer.zero_grad()
            # Forward pass
            torch.cuda.synchronize()
            forward_start = default_timer()
            logits = model(input_batch)
            # Compute loss
            loss = cross_entropy_loss(logits, target_batch)
            # Compute gradients
            loss.backward()
            # Update parameters
            optimizer.step()
            torch.cuda.synchronize()
            optimizer_end = default_timer()
            full_times[i] = optimizer_end - forward_start

        # Time the forward, backward, and optimizer step
        forward_times = np.zeros(time_steps)
        backward_times = np.zeros(time_steps)
        optimizer_times = np.zeros(time_steps)  
        for i in range(time_steps):
            # Clear gradients from the previous step
            optimizer.zero_grad()
            # Forward pass
            torch.cuda.synchronize()
            forward_start = default_timer()
            logits = model(input_batch)
            torch.cuda.synchronize()
            forward_end = default_timer()
            forward_times[i] = forward_end - forward_start
            backward_start = default_timer()
            # Compute loss
            loss = cross_entropy_loss(logits, target_batch)
            # Compute gradients
            loss.backward()
            torch.cuda.synchronize()
            backward_end = default_timer()
            backward_times[i] = backward_end - backward_start
            optimizer_start = default_timer()
            # Update parameters
            optimizer.step()
            torch.cuda.synchronize()
            optimizer_end = default_timer()
            optimizer_times[i] = optimizer_end - optimizer_start

        statistics["forward_mean"] = (forward_times * 1000).mean()
        statistics["forward_std"] = (forward_times * 1000).std()
        statistics["backward_mean"] = (backward_times * 1000).mean()
        statistics["backward_std"] = (backward_times * 1000).std()
        statistics["optimizer_mean"] = (optimizer_times * 1000).mean()
        statistics["optimizer_std"] = (optimizer_times * 1000).std()
        statistics["full_mean"] = (full_times * 1000).mean()
        statistics["full_std"] = (full_times * 1000).std()

    return statistics

    
def main():
    parser = argparse.ArgumentParser(
        description="Benchmark a transformer model."
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["forward", "forward-backward", "full"],
        required=True
    )
    parser.add_argument(
        "--size",
        type=str,
        required=True
    )
    parser.add_argument(
        "--d_model",
        type=int,
        required=True
    )
    parser.add_argument(
        "--d_ff",
        type=int,
        required=True
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        required=True
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        required=True
    )

    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=5
    )

    parser.add_argument(
        "--time_steps",
        type=int,
        default=10
    )

    parser.add_argument(
        "--profile",
        action="store_true",
    )

    parser.add_argument(
        "--context_length",
        type=int,
        default=512
    )

    parser.add_argument(
        "--compiled",
        action="store_true",
    )

    args = parser.parse_args()
    mode = args.mode
    size = args.size
    d_model = args.d_model
    d_ff = args.d_ff
    num_layers = args.num_layers
    num_heads = args.num_heads
    warmup_steps = args.warmup_steps
    time_steps = args.time_steps

    statistics = benchmark(mode, d_model, d_ff, num_layers, num_heads, warmup_steps=warmup_steps, time_steps=time_steps, profile=args.profile, context_length=args.context_length, compiled=args.compiled)

    df = pd.DataFrame([{
        "size": size,
        "context_length": args.context_length,
        "mode": mode,
        **statistics,
    }])

    if args.profile:
        output_dir = Path("profile")
    elif args.compiled:
        output_dir = Path("benchmark_compiled")
    else:
        output_dir = Path("benchmark")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_filepath = output_dir / (
        f"{mode}-{size}"
        f"-ctx{args.context_length}"
        f"-warmup{args.warmup_steps}.tex"
    )
    df.to_latex(
        output_filepath,
        index=False,
        float_format="%.3f",
    )

if __name__ == "__main__":
    main()
        