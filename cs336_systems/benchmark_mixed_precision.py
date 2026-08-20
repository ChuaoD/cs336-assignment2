from __future__ import annotations

from cs336_basics.model import TransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.utils import cross_entropy_loss
   

import torch
from timeit import default_timer
import argparse
import numpy as np
import pandas as pd
import gc
from pathlib import Path
from contextlib import nullcontext

def benchmark(
        d_model: int,
        d_ff: int,
        num_layers: int,
        num_heads: int,
        output_dir: Path,
        batch_size: int = 4,
        context_length: int = 512,
        vocab_size: int = 10000,
        device: str | torch.device = "cuda",
        theta: float = 10000.0,
        warmup_steps: int = 5,
        time_steps: int = 10,
        mixed_precision: bool = False,
        profile_memory: bool = False,
        memory_mode: str = "forward"
):

    device = torch.device(device)
    torch.set_float32_matmul_precision('high')
    
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

    precision = "bf16" if mixed_precision else "fp32"

    model.train()

    statistics = {}

    if profile_memory: 
        snapshot_path = output_dir / (
            f"memory-{d_model}-{precision}"
            f"-ctx{context_length}-{memory_mode}.pickle"
        )
        if memory_mode == "forward":
            # Warm up
            for _ in range(warmup_steps):
                with torch.no_grad():
                    logits = run_forward()
                torch.cuda.synchronize(device)
                del logits
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            # Forward pass
            # Start recording memory history
            torch.cuda.memory._record_memory_history(max_entries=1000000)
            with torch.no_grad():
                logits = run_forward()
            torch.cuda.synchronize()
            # Save a pickle file to be loaded by PyTorch's online tool
            torch.cuda.memory._dump_snapshot(str(snapshot_path))
            # Stop recording history
            torch.cuda.memory._record_memory_history(enabled=None)
            peak_allocated_mib = (
                torch.cuda.max_memory_allocated(device) / 1024**2
            )
            peak_reserved_mib = (
                torch.cuda.max_memory_reserved(device) / 1024**2
            )
            return {
                "peak_allocated_mib": peak_allocated_mib,
                "peak_reserved_mib": peak_reserved_mib,
            }
        
        elif memory_mode == "full":
            # Warm up
            for _ in range(warmup_steps):
                optimizer.zero_grad(set_to_none=True)
                logits = run_forward()
                loss = run_loss(logits)
                loss.backward()
                optimizer.step()
                torch.cuda.synchronize(device)
                del logits, loss
            optimizer.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            # Forward pass
            # Start recording memory history
            torch.cuda.memory._record_memory_history(max_entries=1000000)
            # Forward pass
            logits = run_forward()
            # Compute loss
            loss = run_loss(logits)
            # Compute gradients
            loss.backward()
            # Update parameters
            optimizer.step()
            torch.cuda.synchronize()
            # Save a pickle file to be loaded by PyTorch's online tool
            torch.cuda.memory._dump_snapshot(str(snapshot_path))
            # Stop recording history
            torch.cuda.memory._record_memory_history(enabled=None)
            peak_allocated_mib = (
                torch.cuda.max_memory_allocated(device) / 1024**2
            )
            peak_reserved_mib = (
                torch.cuda.max_memory_reserved(device) / 1024**2
            )
            return {
                "peak_allocated_mib": peak_allocated_mib,
                "peak_reserved_mib": peak_reserved_mib,
            }
    else:
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
        description="Benchmark a Transformer model with mixed precision."
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
        "--context_length",
        type=int,
        default=512
    )

    parser.add_argument(
        "--mixed_precision",
        action="store_true",
        help="Enable CUDA BF16 autocast.",
    )

    parser.add_argument(
        "--profile_memory",
        action="store_true"
    )

    parser.add_argument(
        "--memory_mode",
        choices=["forward", "full"],
        default=None,
    )

    args = parser.parse_args()
    if args.profile_memory and args.memory_mode is None:
        parser.error("--memory_mode is required with --profile_memory")
    size = args.size
    d_model = args.d_model
    d_ff = args.d_ff
    num_layers = args.num_layers
    num_heads = args.num_heads
    warmup_steps = args.warmup_steps
    time_steps = args.time_steps

    if args.profile_memory:
        output_dir = Path("profiling_memory")
    elif args.mixed_precision:
        output_dir = Path("mixed_precision")
    else:
        output_dir = Path("benchmark")
    output_dir.mkdir(parents=True, exist_ok=True)

    statistics = benchmark(d_model, d_ff, num_layers, num_heads, output_dir=output_dir, warmup_steps=warmup_steps, time_steps=time_steps, context_length=args.context_length, mixed_precision=args.mixed_precision, profile_memory=args.profile_memory, memory_mode=args.memory_mode)

    precision = "bf16" if args.mixed_precision else "fp32"
    df = pd.DataFrame([{
        "size": size,
        "context_length": args.context_length,
        "precision": precision,
        **statistics,
    }])
    
    memory_suffix = (
        f"-{args.memory_mode}" if args.profile_memory else ""
    )

    output_filepath = output_dir / (
        f"{size}-{precision}{memory_suffix}"
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

        