from __future__ import annotations

from cs336_basics.model import TransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.utils import cross_entropy_loss
from cs336_systems.ddp import OverlapDDP
from cs336_systems.sharded_optimizer import ShardedOptimizer

import time
import torch
import argparse
import torch.distributed as dist
import torch.multiprocessing as mp

theta = 10000.0
d_model = 2560
d_ff = 10240
num_layers = 32
num_heads = 32
vocab_size = 10000

MB = 1e6
GB = 1e9

def setup(rank, world_size):
    torch.cuda.set_device(rank)

    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        device_id=torch.device(f"cuda:{rank}"),
    )

def cleanup():
    dist.destroy_process_group()

def benchmark_optim(rank, world_size, mode, batch_size, context_length):
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)
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
    ddp_model = OverlapDDP(model)
    ddp_model.train()
    optimizer_config = {
        "lr": 1e-3,
        "betas": (0.9, 0.95),
        "eps": 1e-6,
        "weight_decay": 0.01
    }
    torch.cuda.synchronize()
    if mode == "ord":
        optimizer = AdamW(ddp_model.parameters(), **optimizer_config)
    elif mode == "shard":
        optimizer = ShardedOptimizer(
            ddp_model.parameters(), 
            AdamW, 
            **optimizer_config
        )

    torch.cuda.synchronize()
    after_init = torch.tensor(
        torch.cuda.memory_allocated(device), 
        device=device, dtype=torch.int64
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
    torch.cuda.synchronize()
    # Profile memory
    torch.cuda.reset_peak_memory_stats(device)
    optimizer.zero_grad(set_to_none=True)
    logits = ddp_model(input_batch)
    loss = cross_entropy_loss(logits, target_batch)
    loss.backward()
    ddp_model.finish_gradient_synchronization()
    torch.cuda.synchronize()
    before_step = torch.tensor(
        torch.cuda.memory_allocated(device),
        device=device, dtype=torch.int64
    )
    fwd_bwd_peak  = torch.tensor(
        torch.cuda.max_memory_allocated(device),
        device=device, dtype=torch.int64
    )
    gradient_bytes = torch.tensor(
        get_gradient_bytes(ddp_model),
        device=device, dtype=torch.int64
    )
    torch.cuda.reset_peak_memory_stats(device)
    optimizer.step()
    torch.cuda.synchronize()
    after_step = torch.tensor(
        torch.cuda.memory_allocated(device),
        device=device, dtype=torch.int64
    )
    optim_step_peak = torch.tensor(
        torch.cuda.max_memory_allocated(device), 
        device=device, dtype=torch.int64
    )
    optim_state_bytes = torch.tensor(
        get_optimizer_state_bytes(optimizer), 
        device=device, dtype=torch.int64
    )

    gathered_after_init = [torch.tensor(0, dtype=torch.int64, device=device) for _ in range(world_size)]
    gathered_before_step = [torch.tensor(0, dtype=torch.int64, device=device) for _ in range(world_size)]
    gathered_fwd_bwd_peak = [torch.tensor(0, dtype=torch.int64, device=device) for _ in range(world_size)]
    gathered_after_step = [torch.tensor(0, dtype=torch.int64, device=device) for _ in range(world_size)]
    gathered_optim_step_peak = [torch.tensor(0, dtype=torch.int64, device=device) for _ in range(world_size)]
    gathered_optim_state_bytes = [torch.tensor(0, dtype=torch.int64, device=device) for _ in range(world_size)]
    gathered_gradient_bytes = [torch.tensor(0, dtype=torch.int64, device=device) for _ in range(world_size)]
    dist.all_gather(gathered_after_init, after_init)
    dist.all_gather(gathered_before_step, before_step)
    dist.all_gather(gathered_fwd_bwd_peak, fwd_bwd_peak)
    dist.all_gather(gathered_after_step, after_step)
    dist.all_gather(gathered_optim_step_peak, optim_step_peak)
    dist.all_gather(gathered_optim_state_bytes, optim_state_bytes)
    dist.all_gather(gathered_gradient_bytes, gradient_bytes)

    gathered_after_init = [t.item() / MB for t in gathered_after_init]
    gathered_before_step = [t.item() / MB for t in gathered_before_step]
    gathered_fwd_bwd_peak = [t.item() / MB for t in gathered_fwd_bwd_peak]
    gathered_after_step = [t.item() / MB for t in gathered_after_step]
    gathered_optim_step_peak = [t.item() / MB for t in gathered_optim_step_peak]
    gathered_optim_state_bytes = [t.item() / GB  for t in gathered_optim_state_bytes]
    gathered_gradient_bytes = [t.item() / GB for t in gathered_gradient_bytes]

    if rank == 0:
        parameter_gb = get_parameter_bytes(ddp_model) / GB
        print(
            f"mode={mode}, "
            f"world_size={world_size}, "
            f"local batch_size={batch_size}, "
            f"context_length={context_length}"
        )
        print(
            "\nMemory",
            f"\nParameters (GB): {parameter_gb}",
            f"\nGradients (GB): {gathered_gradient_bytes}",
            f"\nOptimizer states (GB): {gathered_optim_state_bytes}",
        )
        print(
            "\nAfter initialization (MB):\n",
            gathered_after_init,
            "\nBefore optimizer step (MB):\n",
            gathered_before_step,
            "\nPeak memory in forward-backward pass:\n",
            gathered_fwd_bwd_peak,
            "\nAfter optimizer step (MB):\n",
            gathered_after_step,
            "\nPeak memory in optimizer step (MB):\n",
            gathered_optim_step_peak
        )

    # Benchmark
    # Warm up
    for _ in range(5):
        optimizer.zero_grad(set_to_none=True)
        logits = ddp_model(input_batch)
        loss = cross_entropy_loss(logits, target_batch)
        loss.backward()
        ddp_model.finish_gradient_synchronization()
        optimizer.step()
    torch.cuda.synchronize()
    n_iters = 10
    total_time = 0.0
    # Time full training step
    for i in range(n_iters):
        # Clear gradients from the previous step
        optimizer.zero_grad()
        torch.cuda.synchronize()
        dist.barrier()
        forward_start = time.perf_counter()
        # Forward pass
        logits = ddp_model(input_batch)
        # Compute loss
        loss = cross_entropy_loss(logits, target_batch)
        # Compute gradients
        loss.backward()
        ddp_model.finish_gradient_synchronization()
        # Update parameters
        optimizer.step()
        torch.cuda.synchronize()
        optimizer_end = time.perf_counter()
        total_time += (optimizer_end - forward_start)
    total_time_mean = torch.tensor(total_time / n_iters, device=device, dtype=torch.float32)
    gathered_total_time_mean  = [torch.tensor(0.0, device=device) for _ in range(world_size)]
    dist.all_gather(gathered_total_time_mean, total_time_mean)
    gathered_total_time_mean = [t.item() * 1000 for t in gathered_total_time_mean]
    if rank == 0:
        print(
            "\nTotal time (ms): ",
            gathered_total_time_mean
        )

    cleanup()

def get_parameter_bytes(model: torch.nn.Module) -> int:
    return sum(
        parameter.numel() * parameter.element_size()
        for parameter in model.parameters()
    )
def get_gradient_bytes(model: torch.nn.Module) -> int:
    return sum(
        parameter.grad.numel() * parameter.grad.element_size()
        for parameter in model.parameters()
        if parameter.grad is not None
    )
def get_optimizer_state_bytes(
    optimizer: torch.optim.Optimizer,
) -> int:
    state_owner = (
        optimizer.inner
        if isinstance(optimizer, ShardedOptimizer)
        else optimizer
    )

    total_bytes = 0

    for parameter_state in state_owner.state.values():
        for value in parameter_state.values():
            if torch.is_tensor(value):
                total_bytes += value.numel() * value.element_size()

    return total_bytes

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["ord", "shard"],
        required=True
    )
  
    args = parser.parse_args()
    world_size = 2
    batch_size = 2
    context_length = 512
    mp.spawn(
        fn=benchmark_optim, 
        args=(world_size, args.mode, batch_size, context_length), 
        nprocs=world_size, 
        join=True
    )
    
if __name__ == "__main__":
    main()
        