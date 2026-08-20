import os
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    torch.cuda.set_device(rank)

    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        device_id=torch.device(f"cuda:{rank}"),
    )

def cleanup():
    dist.destroy_process_group()

def distributed_comm(rank, world_size, numel):
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")
    data = torch.zeros(numel, dtype=torch.float32, device=device)
    # Warm up
    for _ in range(5):
        dist.all_reduce(data, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    dist.barrier()
    # Benchmark
    n_iters = 100
    start = time.perf_counter()
    for _ in range(n_iters):
        dist.all_reduce(data, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - start) / n_iters * 1000
    gathered_latencies = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_latencies, latency_ms)
    if rank == 0:
        print(
            f"world_size={world_size}, "
            f"numel={numel}, "
            f"size={numel * 4 / 1e6:.1f} MB"
        )
        print(
            "latencies (ms):",
            gathered_latencies,
        )
    cleanup()
    

def main():
    for world_size in [1, 2, 4]:
        for numel in [250_000 * (10 ** i) for i in range(0, 4)]:
            mp.spawn(fn=distributed_comm, args=(world_size, numel), nprocs=world_size, join=True)
    
if __name__ == "__main__":
    main()