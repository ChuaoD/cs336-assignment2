from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from cs336_basics.optimizer import AdamW
from cs336_systems.fsdp import FSDP
from cs336_systems.model_flash import TransformerLM


@dataclass(frozen=True)
class BenchmarkConfig:
    world_size: int
    global_batch_size: int
    context_length: int
    vocab_size: int
    d_model: int
    d_ff: int
    num_layers: int
    num_heads: int
    warmup_steps: int
    measured_steps: int
    activation_checkpointing: bool


def setup(rank: int, world_size: int) -> torch.device:
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12390")
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        device_id=device,
    )
    return device


def cleanup() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def benchmark_worker(rank: int, config: BenchmarkConfig) -> None:
    device = setup(rank, config.world_size)

    try:
        if config.global_batch_size % config.world_size != 0:
            raise ValueError("global_batch_size must be divisible by world_size")

        local_batch_size = config.global_batch_size // config.world_size
        torch.manual_seed(1234 + rank)
        input_ids = torch.randint(
            config.vocab_size,
            (local_batch_size, config.context_length),
            dtype=torch.long,
            device=device,
        )
        targets = torch.randint(
            config.vocab_size,
            (local_batch_size, config.context_length),
            dtype=torch.long,
            device=device,
        )

        model = TransformerLM(
            vocab_size=config.vocab_size,
            context_length=config.context_length,
            num_layers=config.num_layers,
            d_model=config.d_model,
            num_heads=config.num_heads,
            d_ff=config.d_ff,
            activation_checkpointing=config.activation_checkpointing,
            theta=10_000.0,
            device=device,
        )
        fsdp_model = FSDP(model, compute_dtype=torch.bfloat16)
        optimizer = AdamW(fsdp_model.parameters())
        fsdp_model.train()

        def train_step() -> torch.Tensor:
            optimizer.zero_grad(set_to_none=True)
            loss_sum = fsdp_model(input_ids, targets=targets)
            loss = loss_sum / targets.numel()
            loss.backward()
            fsdp_model.finish_gradient_synchronization()
            optimizer.step()
            return loss.detach()

        # Warmup compiles Triton kernels and initializes AdamW state.
        last_loss = torch.zeros((), dtype=torch.float32, device=device)
        for _ in range(config.warmup_steps):
            last_loss = train_step()

        optimizer.zero_grad(set_to_none=True)
        dist.barrier()
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        baseline_allocated = torch.cuda.memory_allocated(device)

        start = time.perf_counter()
        for _ in range(config.measured_steps):
            last_loss = train_step()
        torch.cuda.synchronize(device)
        elapsed_seconds = time.perf_counter() - start

        local_average_ms = elapsed_seconds * 1_000 / config.measured_steps
        local_peak_allocated = torch.cuda.max_memory_allocated(device)
        local_peak_reserved = torch.cuda.max_memory_reserved(device)

        # Distributed step latency is determined by the slowest rank.
        maximum_metrics = torch.tensor(
            [
                local_average_ms,
                float(baseline_allocated),
                float(local_peak_allocated),
                float(local_peak_reserved),
            ],
            dtype=torch.float64,
            device=device,
        )
        dist.all_reduce(maximum_metrics, op=dist.ReduceOp.MAX)

        mean_loss = last_loss.float().clone()
        dist.all_reduce(mean_loss, op=dist.ReduceOp.SUM)
        mean_loss /= config.world_size

        if rank == 0:
            max_step_ms, max_baseline, max_allocated, max_reserved = (
                maximum_metrics.tolist()
            )
            global_tokens = config.global_batch_size * config.context_length
            tokens_per_second = global_tokens / (max_step_ms / 1_000)
            gib = 1024**3

            print(f"world_size={config.world_size}")
            print(f"global_batch_size={config.global_batch_size}")
            print(f"local_batch_size={local_batch_size}")
            print(f"context_length={config.context_length}")
            print(f"num_layers={config.num_layers}")
            print(
                "activation_checkpointing="
                f"{config.activation_checkpointing}"
            )
            print(f"warmup_steps={config.warmup_steps}")
            print(f"measured_steps={config.measured_steps}")
            print(f"max_rank_step_ms={max_step_ms:.3f}")
            print(f"global_tokens_per_second={tokens_per_second:.2f}")
            print(f"max_rank_baseline_allocated_gib={max_baseline / gib:.3f}")
            print(f"max_rank_peak_allocated_gib={max_allocated / gib:.3f}")
            print(f"max_rank_peak_reserved_gib={max_reserved / gib:.3f}")
            print(f"final_distributed_mean_loss={mean_loss.item():.6f}")
    finally:
        cleanup()


def parse_args() -> BenchmarkConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--global-batch-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=32_768)
    parser.add_argument("--vocab-size", type=int, default=151_936)
    parser.add_argument("--d-model", type=int, default=4_096)
    parser.add_argument("--d-ff", type=int, default=11_008)
    parser.add_argument("--num-layers", type=int, default=34)
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--measured-steps", type=int, default=3)
    parser.add_argument(
        "--activation-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()
    return BenchmarkConfig(
        world_size=args.world_size,
        global_batch_size=args.global_batch_size,
        context_length=args.context_length,
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        d_ff=args.d_ff,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        warmup_steps=args.warmup_steps,
        measured_steps=args.measured_steps,
        activation_checkpointing=args.activation_checkpointing,
    )


def main() -> None:
    config = parse_args()
    if config.world_size <= 0:
        raise ValueError("world_size must be positive")
    if config.warmup_steps < 1:
        raise ValueError("warmup_steps must be at least one")
    if config.measured_steps < 1:
        raise ValueError("measured_steps must be at least one")
    if torch.cuda.device_count() < config.world_size:
        raise RuntimeError(
            f"Requested {config.world_size} GPUs, but only "
            f"{torch.cuda.device_count()} are visible"
        )

    mp.spawn(
        benchmark_worker,
        args=(config,),
        nprocs=config.world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
