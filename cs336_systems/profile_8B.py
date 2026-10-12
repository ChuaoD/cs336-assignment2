from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import torch
import torch.cuda.nvtx as nvtx
import torch.distributed as dist
import torch.multiprocessing as mp

from cs336_basics.optimizer import AdamW
from cs336_systems.benchmark_8B import cleanup, setup
from cs336_systems.fsdp import FSDP
from cs336_systems.fused_adamw import FusedAdamW
from cs336_systems.model_flash import TransformerLM


@dataclass(frozen=True)
class ProfileConfig:
    world_size: int
    global_batch_size: int
    context_length: int
    vocab_size: int
    d_model: int
    d_ff: int
    num_layers: int
    num_heads: int
    warmup_steps: int
    profiled_steps: int
    optimizer: str
    activation_checkpointing: bool


def profile_worker(rank: int, config: ProfileConfig) -> None:
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
        # enable_nvtx labels each layer's forward and every all-gather launch
        # and wait, which makes the FSDP schedule readable in the timeline.
        fsdp_model = FSDP(model, compute_dtype=torch.bfloat16, enable_nvtx=True)
        optimizer_cls = FusedAdamW if config.optimizer == "fused-adamw" else AdamW
        optimizer = optimizer_cls(fsdp_model.parameters())
        fsdp_model.train()

        def train_step() -> torch.Tensor:
            optimizer.zero_grad(set_to_none=True)
            # The fused LM-head loss runs inside the model's forward pass.
            with nvtx.range("forward"):
                loss_sum = fsdp_model(input_ids, targets=targets)
                loss = loss_sum / targets.numel()
            # Recomputation and the FlashAttention and fused-loss backward
            # kernels carry their own NVTX ranges inside this one.
            with nvtx.range("backward"):
                loss.backward()
            with nvtx.range("finish_gradient_synchronization"):
                fsdp_model.finish_gradient_synchronization()
            with nvtx.range("optimizer"):
                optimizer.step()
            return loss.detach()

        # Warmup compiles Triton kernels, traces FSDP's execution order, and
        # initializes AdamW state before the capture range opens.
        last_loss = torch.zeros((), dtype=torch.float32, device=device)
        for _ in range(config.warmup_steps):
            last_loss = train_step()
        torch.cuda.synchronize(device)

        step_ms = []
        dist.barrier()
        with nvtx.range("profile"):
            for _ in range(config.profiled_steps):
                # Align ranks outside the step range so the barrier is not
                # counted as step time.
                dist.barrier()
                torch.cuda.synchronize(device)
                start = time.perf_counter()
                with nvtx.range("step"):
                    last_loss = train_step()
                    torch.cuda.synchronize(device)
                step_ms.append((time.perf_counter() - start) * 1_000)

        # Distributed step latency is determined by the slowest rank.
        max_rank_step_ms = torch.tensor(step_ms, dtype=torch.float64, device=device)
        dist.all_reduce(max_rank_step_ms, op=dist.ReduceOp.MAX)

        mean_loss = last_loss.float().clone()
        dist.all_reduce(mean_loss, op=dist.ReduceOp.SUM)
        mean_loss /= config.world_size

        if rank == 0:
            print("benchmark=8B_nsys_profile")
            print(f"world_size={config.world_size}")
            print(f"optimizer={optimizer_cls.__module__}.{optimizer_cls.__name__}")
            print(f"global_batch_size={config.global_batch_size}")
            print(f"local_batch_size={local_batch_size}")
            print(f"context_length={config.context_length}")
            print(f"num_layers={config.num_layers}")
            print(
                "activation_checkpointing="
                f"{config.activation_checkpointing}"
            )
            print(f"warmup_steps={config.warmup_steps}")
            print(f"profiled_steps={config.profiled_steps}")
            # Profiled steps include tracing overhead; use benchmark_8B.py for
            # the headline step time and this trace for proportions.
            print(
                "max_rank_profiled_step_ms="
                + ",".join(f"{value:.3f}" for value in max_rank_step_ms.tolist())
            )
            print(f"final_distributed_mean_loss={mean_loss.item():.6f}")
    finally:
        cleanup()


def parse_args() -> ProfileConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--global-batch-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=32_768)
    parser.add_argument("--vocab-size", type=int, default=151_936)
    parser.add_argument("--d-model", type=int, default=4_096)
    parser.add_argument("--d-ff", type=int, default=11_008)
    parser.add_argument("--num-layers", type=int, default=34)
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--profiled-steps", type=int, default=2)
    parser.add_argument(
        "--optimizer", choices=["adamw", "fused-adamw"], default="fused-adamw"
    )
    parser.add_argument(
        "--activation-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()
    return ProfileConfig(
        world_size=args.world_size,
        global_batch_size=args.global_batch_size,
        context_length=args.context_length,
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        d_ff=args.d_ff,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        warmup_steps=args.warmup_steps,
        profiled_steps=args.profiled_steps,
        optimizer=args.optimizer,
        activation_checkpointing=args.activation_checkpointing,
    )


def main() -> None:
    config = parse_args()
    if config.world_size <= 0:
        raise ValueError("world_size must be positive")
    # The first step traces FSDP's execution order and creates optimizer state;
    # a second step lets prefetch buffers and the caching allocator settle.
    if config.warmup_steps < 2:
        raise ValueError("warmup_steps must be at least two")
    if config.profiled_steps < 1:
        raise ValueError("profiled_steps must be at least one")
    if torch.cuda.device_count() < config.world_size:
        raise RuntimeError(
            f"Requested {config.world_size} GPUs, but only "
            f"{torch.cuda.device_count()} are visible"
        )

    mp.spawn(
        profile_worker,
        args=(config,),
        nprocs=config.world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
