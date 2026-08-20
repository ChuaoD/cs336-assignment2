from __future__ import annotations

import torch
from timeit import default_timer
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import gc

from cs336_basics.utils import attention

HEAD_DIMS = [16, 32, 64, 128]
SEQ_LENGTHS = [256, 1024, 4096, 8192, 16384]
BATCH_SIZE = 8
MIB = 1024**2

def benchmark_attention(
        d_model: int,
        seq_len: int,
        batch_size: int = 8,
        device: str | torch.device = "cuda",
        warmup_steps: int = 5,
        time_steps: int = 100,
        compiled: bool = False
):
    mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device))
    torch.set_float32_matmul_precision('high')
    Q = torch.randn(
        batch_size,
        seq_len,
        d_model,
        device=device,
        requires_grad=True,
    )
    K = torch.randn_like(Q, requires_grad=True)
    V = torch.randn_like(Q, requires_grad=True)
    grad_output = torch.randn_like(Q)

    if compiled:
        attention_fn = torch.compile(attention)
    else:
        attention_fn = attention

    # Warm up
    for _ in range(warmup_steps):
        Q.grad = K.grad = V.grad = None
        output = attention_fn(Q, K, V, mask)
        torch.cuda.synchronize(device)
        del output

    forward_times = np.zeros(time_steps)    
    # Time training forward pass
    for i in range(time_steps):
        Q.grad = K.grad = V.grad = None
        torch.cuda.synchronize(device)
        forward_start = default_timer()
        output = attention_fn(Q, K, V, mask)
        torch.cuda.synchronize(device)
        forward_end = default_timer()
        forward_times[i] = forward_end - forward_start
        del output

    # Measure memory
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    before_forward_bytes = torch.cuda.memory_allocated(device)
    output = attention_fn(Q, K, V, mask)
    torch.cuda.synchronize(device)
    before_backward_bytes = torch.cuda.memory_allocated(device)
    saved_for_backward_mib = (before_backward_bytes - before_forward_bytes) / MIB
    del output
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)

    # Warm up for backward pass
    for _ in range(warmup_steps):
        Q.grad = K.grad = V.grad = None

        output = attention_fn(Q, K, V, mask)
        output.backward(grad_output)

        torch.cuda.synchronize(device)
        del output
    # Time backward pass
    backward_times = np.zeros(time_steps)
    for i in range(time_steps):
        Q.grad = K.grad = V.grad = None
        output = attention_fn(Q, K, V, mask)
        torch.cuda.synchronize(device)
        backward_start = default_timer()
        output.backward(grad_output)
        torch.cuda.synchronize(device)
        backward_end = default_timer()
        backward_times[i] = backward_end - backward_start
        del output 
    
    statistics = {
        "forward mean": (forward_times * 1000).mean(),
        "forward std": (forward_times * 1000).std(),
        "backward mean": (backward_times * 1000).mean(),
        "backward std": (backward_times * 1000).std(),
        "saved for backward memory mib": saved_for_backward_mib
    }
    return statistics

    
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--d_model",
        type=int,
        required=True
    )

    parser.add_argument(
        "--seq_len",
        type=int,
        default=512
    )

    parser.add_argument(
        "--compiled",
        action="store_true"
    )
    
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=5
    )

    parser.add_argument(
        "--time_steps",
        type=int,
        default=100
    )

    args = parser.parse_args()
    d_model = args.d_model
    seq_len = args.seq_len
    warmup_steps = args.warmup_steps
    time_steps = args.time_steps

    statistics = benchmark_attention(d_model, seq_len, warmup_steps=warmup_steps, time_steps=time_steps, compiled=args.compiled)

    df = pd.DataFrame([{
        "d_model": d_model,
        "seq_len": seq_len,
        **statistics,
    }])

    output_dir = Path("benchmark_attention")
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = "compiled" if args.compiled else ""
  
    output_filepath = output_dir / (
        f"d{d_model}-ctx{args.seq_len}{suffix}.tex"
    )
    df.to_latex(
        output_filepath,
        index=False,
        float_format="%.3f",
    )

if __name__ == "__main__":
    main()
