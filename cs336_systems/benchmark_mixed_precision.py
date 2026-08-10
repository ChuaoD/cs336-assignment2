from __future__ import annotations

from cs336_basics.model import TransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.utils import cross_entropy_loss
   

import torch
from timeit import default_timer
import argparse
import numpy as np
import pandas as pd
import math
from pathlib import Path

import cs336_basics.model as basics_model
import torch.cuda.nvtx as nvtx
from cs336_basics.utils import softmax
from einops import einsum

from contextlib import nullcontext

@nvtx.range("scaled dot product attention")
def annotated_attention(Q, K, V, mask=None):
    d_k = K.shape[-1]

    with nvtx.range("computing attention scores"):
        score = (
            einsum(
                Q,
                K,
                "... num_q d_k, ... num_k d_k -> ... num_q num_k",
            )
            / math.sqrt(d_k)
        )

        if mask is not None:
            score = score.masked_fill(~mask, value=float("-inf"))

    with nvtx.range("computing softmax"):
        probabilities = softmax(score, dim=-1)

    with nvtx.range("final matmul"):
        output = einsum(
            probabilities,
            V,
            "... num_q num_k, ... num_k d_v -> ... num_q d_v",
        )

    return output

def benchmark(
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
        mixed_precision: bool = False,
):

    device = torch.device(device)

    if profile:
        basics_model.attention = annotated_attention

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

    def precision_context():
        if mixed_precision:
            return torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
            )
        return nullcontext()


    def run_forward():
        with precision_context():
            return model(input_batch)


    def run_loss(logits):
        with precision_context():
            return cross_entropy_loss(logits, target_batch)

    model.train()

    statistics = {}

    # Warm up
    for _ in range(warmup_steps):
        optimizer.zero_grad(set_to_none=True)
        logits = run_forward()
        loss = run_loss(logits)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize(device)
        del logits

    # Time the forward and backward step
    forward_times = np.zeros(time_steps)
    backward_times = np.zeros(time_steps)
    for i in range(time_steps):
        # Clear gradients from the previous step
        optimizer.zero_grad()
        # Forward pass
        torch.cuda.synchronize()
        forward_start = default_timer()
        logits = run_forward()
        torch.cuda.synchronize()
        forward_end = default_timer()
        forward_times[i] = forward_end - forward_start
        backward_start = default_timer()
        # Compute loss
        loss = run_loss(logits)
        # Compute gradients
        loss.backward()
        torch.cuda.synchronize()
        backward_end = default_timer()
        backward_times[i] = backward_end - backward_start

    statistics["forward_mean"] = (forward_times * 1000).mean()
    statistics["forward_std"] = (forward_times * 1000).std()
    statistics["backward_mean"] = (backward_times * 1000).mean()
    statistics["backward_std"] = (backward_times * 1000).std()
    return statistics

    
def main():
    parser = argparse.ArgumentParser(
        description="Train a language model."
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
        "--mixed_precision",
        action="store_true",
        help="Enable CUDA BF16 autocast.",
    )


    args = parser.parse_args()
    size = args.size
    d_model = args.d_model
    d_ff = args.d_ff
    num_layers = args.num_layers
    num_heads = args.num_heads
    warmup_steps = args.warmup_steps
    time_steps = args.time_steps

    statistics = benchmark(d_model, d_ff, num_layers, num_heads, warmup_steps=warmup_steps, time_steps=time_steps, profile=args.profile, context_length=args.context_length, mixed_precision=args.mixed_precision)

    precision = "bf16" if args.mixed_precision else "fp32"
    df = pd.DataFrame([{
        "size": size,
        "context_length": args.context_length,
        "precision": precision,
        **statistics,
    }])
    if args.profile:
        output_dir = Path("profile")
    elif args.mixed_precision:
        output_dir = Path("mixed_precision")
    else:
        output_dir = Path("benchmark")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_filepath = output_dir / (
        f"{size}-{precision}"
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

        