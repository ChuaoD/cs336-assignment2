from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import triton.runtime as triton_runtime

from cs336_basics.optimizer import AdamW
from cs336_systems.fsdp import FSDP
from cs336_systems.fused_adamw import FusedAdamW
from cs336_systems.model_flash import TransformerLM


@dataclass(frozen=True)
class Section9Config:
    world_size: int = 2
    global_batch_size: int = 2
    context_length: int = 32_768
    vocab_size: int = 151_936
    d_model: int = 4_096
    d_ff: int = 11_008
    num_layers: int = 34
    num_heads: int = 32
    warmup_ms: int = 10_000
    rep_ms: int = 30_000
    activation_checkpointing: bool = True
    optimizer: str = "fused-adamw"


def _setup(rank: int, world_size: int) -> torch.device:
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12391")
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        device_id=device,
    )
    return device


def _cleanup() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def _section9_worker(rank: int, config: Section9Config) -> None:
    device = _setup(rank, config.world_size)

    try:
        if config.world_size != 2:
            raise ValueError("The Section 9 benchmark requires exactly two ranks")
        if config.global_batch_size != 2:
            raise ValueError("The Section 9 benchmark requires global batch size two")

        local_batch_size = config.global_batch_size // config.world_size

        # Every rank must initialize the same logical model before FSDP shards it.
        torch.manual_seed(0)
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
        optimizer_cls = FusedAdamW if config.optimizer == "fused-adamw" else AdamW
        optimizer = optimizer_cls(fsdp_model.parameters())
        fsdp_model.train()

        # Each rank owns a different sample, giving global batch size two.
        data_generator = torch.Generator(device=device)
        data_generator.manual_seed(1_234 + rank)
        labels = torch.randint(
            config.vocab_size,
            (local_batch_size, config.context_length),
            dtype=torch.long,
            device=device,
            generator=data_generator,
        )
        targets = torch.randint(
            config.vocab_size,
            (local_batch_size, config.context_length),
            dtype=torch.long,
            device=device,
            generator=data_generator,
        )

        def train_step() -> torch.Tensor:
            optimizer.zero_grad(set_to_none=True)
            local_loss_sum = fsdp_model(labels, targets=targets)

            # FSDP averages gradients across ranks. Scaling each local summed
            # loss recovers the gradient of the global summed loss in the PDF.
            objective = local_loss_sum * config.world_size
            objective.backward()
            fsdp_model.finish_gradient_synchronization()
            optimizer.step()
            return (local_loss_sum / targets.numel()).detach()

        # Match triton.testing.do_bench: one initial call, then five calls to
        # estimate step duration. All ranks execute exactly the same calls.
        last_mean_loss = train_step()
        torch.cuda.synchronize(device)
        cache = triton_runtime.driver.active.get_empty_cache_for_benchmark()

        dist.barrier()
        estimate_start = torch.cuda.Event(enable_timing=True)
        estimate_end = torch.cuda.Event(enable_timing=True)
        estimate_start.record()
        for _ in range(5):
            triton_runtime.driver.active.clear_cache(cache)
            last_mean_loss = train_step()
        estimate_end.record()
        torch.cuda.synchronize(device)

        estimate_ms = torch.tensor(
            estimate_start.elapsed_time(estimate_end) / 5,
            dtype=torch.float64,
            device=device,
        )
        dist.all_reduce(estimate_ms, op=dist.ReduceOp.MAX)

        # Triton interprets warmup and rep as millisecond budgets and floors
        # their ratios, with at least one step in each phase.
        warmup_steps = max(1, int(config.warmup_ms / estimate_ms.item()))
        measured_steps = max(1, int(config.rep_ms / estimate_ms.item()))

        for _ in range(warmup_steps):
            last_mean_loss = train_step()

        dist.barrier()
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        baseline_allocated = torch.cuda.memory_allocated(device)

        start_events = [
            torch.cuda.Event(enable_timing=True) for _ in range(measured_steps)
        ]
        end_events = [
            torch.cuda.Event(enable_timing=True) for _ in range(measured_steps)
        ]

        for step in range(measured_steps):
            triton_runtime.driver.active.clear_cache(cache)
            start_events[step].record()
            last_mean_loss = train_step()
            end_events[step].record()

        torch.cuda.synchronize(device)
        rank_step_times = torch.tensor(
            [start.elapsed_time(end) for start, end in zip(start_events, end_events)],
            dtype=torch.float64,
            device=device,
        )

        # A distributed step completes only when its slowest rank completes.
        dist.all_reduce(rank_step_times, op=dist.ReduceOp.MAX)

        memory_metrics = torch.tensor(
            [
                float(baseline_allocated),
                float(torch.cuda.max_memory_allocated(device)),
                float(torch.cuda.max_memory_reserved(device)),
            ],
            dtype=torch.float64,
            device=device,
        )
        dist.all_reduce(memory_metrics, op=dist.ReduceOp.MAX)

        distributed_mean_loss = last_mean_loss.float().clone()
        dist.all_reduce(distributed_mean_loss, op=dist.ReduceOp.SUM)
        distributed_mean_loss /= config.world_size

        if rank == 0:
            timing_ms = rank_step_times.mean().item()
            tokens_per_second = (
                config.global_batch_size
                * config.context_length
                / (timing_ms / 1_000)
            )
            baseline, peak_allocated, peak_reserved = memory_metrics.tolist()
            gib = 1024**3

            print("benchmark=section9_distributed_do_bench_equivalent")
            print(f"optimizer={optimizer_cls.__module__}.{optimizer_cls.__name__}")
            print(f"world_size={config.world_size}")
            print(f"global_batch_size={config.global_batch_size}")
            print(f"local_batch_size={local_batch_size}")
            print(f"context_length={config.context_length}")
            print(f"vocab_size={config.vocab_size}")
            print(f"d_model={config.d_model}")
            print(f"d_ff={config.d_ff}")
            print(f"num_layers={config.num_layers}")
            print(f"num_heads={config.num_heads}")
            print("compute_dtype=torch.bfloat16")
            print(
                "activation_checkpointing="
                f"{config.activation_checkpointing}"
            )
            print(f"requested_warmup_ms={config.warmup_ms}")
            print(f"requested_rep_ms={config.rep_ms}")
            print(f"estimated_step_ms={estimate_ms.item():.3f}")
            print(f"derived_warmup_steps={warmup_steps}")
            print(f"derived_measured_steps={measured_steps}")
            print(
                "max_rank_step_times_ms="
                + ",".join(f"{value:.3f}" for value in rank_step_times.tolist())
            )
            print(f"timing_results_ms={timing_ms:.3f}")
            print(f"global_tokens_per_second={tokens_per_second:.2f}")
            print(f"max_rank_baseline_allocated_gib={baseline / gib:.3f}")
            print(f"max_rank_peak_allocated_gib={peak_allocated / gib:.3f}")
            print(f"max_rank_peak_reserved_gib={peak_reserved / gib:.3f}")
            print(
                "final_distributed_mean_loss="
                f"{distributed_mean_loss.item():.6f}"
            )
    finally:
        _cleanup()


def _parse_args() -> Section9Config:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup-ms", type=int, default=10_000)
    parser.add_argument("--rep-ms", type=int, default=30_000)
    parser.add_argument(
        "--optimizer", choices=["adamw", "fused-adamw"], default="fused-adamw"
    )
    parser.add_argument(
        "--activation-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    if args.warmup_ms <= 0:
        raise ValueError("warmup_ms must be positive")
    if args.rep_ms <= 0:
        raise ValueError("rep_ms must be positive")

    return Section9Config(
        optimizer=args.optimizer,
        warmup_ms=args.warmup_ms,
        rep_ms=args.rep_ms,
        activation_checkpointing=args.activation_checkpointing,
    )


def main() -> None:
    config = _parse_args()
    if torch.cuda.device_count() != config.world_size:
        raise RuntimeError(
            f"Expected exactly {config.world_size} visible GPUs, got "
            f"{torch.cuda.device_count()}"
        )

    mp.spawn(
        _section9_worker,
        args=(config,),
        nprocs=config.world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
