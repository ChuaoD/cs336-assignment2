from __future__ import annotations

from cs336_basics.model import TransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.utils import cross_entropy_loss
from cs336_systems.ddp import NaiveDDP, FlatDDP, OverlapDDP

import torch
import time
import os
import argparse
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.cuda.nvtx as nvtx

theta = 10000.0
d_model = 2560
d_ff = 10240
num_layers = 32
num_heads = 32
vocab_size = 10000

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

def benchmark_ddp(rank, world_size, mode, profile, batch_size, context_length):
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")
    
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
    time_comm = True
    if mode == "naive":
        ddp_model = NaiveDDP(model)
    elif mode == "flat":
        ddp_model = FlatDDP(model)
    elif mode == "overlap":
        ddp_model = OverlapDDP(model)
        time_comm = False
    
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
    
    optimizer = AdamW(ddp_model.parameters(), **optimizer_config)

    ddp_model.train()

    # Warm up
    for _ in range(5):
        optimizer.zero_grad(set_to_none=True)
        logits = ddp_model(input_batch)
        loss = cross_entropy_loss(logits, target_batch)
        loss.backward()
        ddp_model.finish_gradient_synchronization()
        optimizer.step()
    torch.cuda.synchronize()

    if profile:
        optimizer.zero_grad(set_to_none=True)
        dist.barrier()

        with nvtx.range("profile"):
            with nvtx.range("forward"):
                logits = ddp_model(input_batch)

            with nvtx.range("loss"):
                loss = cross_entropy_loss(logits, target_batch)

            with nvtx.range("backward"):
                loss.backward()

            with nvtx.range("finish_gradient_synchronization"):
                ddp_model.finish_gradient_synchronization()

            with nvtx.range("optimizer"):
                optimizer.step()
            torch.cuda.synchronize()
    else:
        # Benchmark
        n_iters = 10
        total_times = np.zeros(n_iters)
        comm_times = np.zeros(n_iters)
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
            total_times[i] = (optimizer_end - forward_start) * 1000
        total_time_mean = torch.tensor(total_times.mean(), device=device, dtype=torch.float32)
        gathered_total_mean  = [torch.tensor(0.0, device=device) for _ in range(world_size)]
        dist.all_gather(gathered_total_mean, total_time_mean)
        if rank == 0:
            print(
                f"mode={mode}, "
                f"world_size={world_size}, "
                f"batch_size={batch_size}, "
                f"context_length={context_length}"
            )
            print(
                "total time (ms): ",
                gathered_total_mean
            )
        if time_comm:
            # Time communication  
            for i in range(n_iters):
                # Clear gradients from the previous step
                optimizer.zero_grad()
                # Forward pass
                logits = ddp_model(input_batch)
                # Compute loss
                loss = cross_entropy_loss(logits, target_batch)
                # Compute gradients
                loss.backward()
                torch.cuda.synchronize()
                dist.barrier()  
                comm_start = time.perf_counter()
                ddp_model.finish_gradient_synchronization()
                torch.cuda.synchronize()
                comm_end = time.perf_counter()
                # Update parameters
                optimizer.step()
                comm_times[i] = (comm_end - comm_start) * 1000
        
            comm_time_mean = torch.tensor(comm_times.mean(), device=device, dtype=torch.float32)
            comm_proportions = comm_time_mean / total_time_mean
            gathered_comm_mean  = [torch.tensor(0.0, device=device) for _ in range(world_size)]
            gathered_comm_prop  = [torch.tensor(0.0, device=device) for _ in range(world_size)]
            dist.all_gather(gathered_comm_mean, comm_time_mean)
            dist.all_gather(gathered_comm_prop, comm_proportions)
            if rank == 0:
                print(
                    "comm time (ms): ",
                    gathered_comm_mean,
                    "comm proportion: ",
                    gathered_comm_prop
                )
    cleanup()
    

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["naive", "flat", "overlap"],
        required=True
    )
    parser.add_argument(
        "--world_size",
        type=int,
        default=2
    )
    parser.add_argument(
        "--profile",
        action="store_true",
    )
    args = parser.parse_args()
    world_size = args.world_size
    batch_size = 2
    context_length = 512
    mp.spawn(
        fn=benchmark_ddp, 
        args=(world_size, args.mode, args.profile, batch_size, context_length), 
        nprocs=world_size, 
        join=True
    )
    
if __name__ == "__main__":
    main()
        