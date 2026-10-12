# Benchmark results

## Experiment configurations

| Experiment | Hardware | Workload | Precision | Implementation |
| --- | --- | --- | --- | --- |
| Triton attention | 1 × A100 80G | Batch 1; head dimension 128; sequence lengths 1,024–32,768; causal | BF16 | [flash_benchmarking.py](../cs336_systems/flash_benchmarking.py) |
| DDP variants | 2 or 4 × A100 80G | XL model; local batch 2; context 512 | FP32 | [benchmark_ddp.py](../cs336_systems/benchmark_ddp.py) |
| Optimizer-state sharding | 2 × A100 80G | XL model; local batch 2; context 512 | FP32 | [benchmark_sharded_optim.py](../cs336_systems/benchmark_sharded_optim.py) |
| FSDP | 2 × A100 80G | XL model; local batch 2; context 512 | FP16 compute; FP32 master shards | [benchmark_fsdp.py](../cs336_systems/benchmark_fsdp.py) |
| 8B attention tiles and TF32 loss | 2 × B200 | Global batch 2; context 32,768; 26 of 34 layers recomputed | BF16 compute; FP32 master shards | [8B configuration and results](section9/README.md) |
| 8B training step, profile | 4 × A100 80G | Global batch 4; context 32,768; 32 of 34 layers recomputed | BF16 compute; FP32 master shards | [benchmark_8B.py](../cs336_systems/benchmark_8B.py) · [profile_8B.py](../cs336_systems/profile_8B.py) · [results](#8b-training-step-on-four-a100-gpus) |
| 8B components, tile sweep | 1 × A100 80G | One rank's shapes: batch 1, context 32,768, 32 heads | BF16 (optimizer FP32) | [benchmark_8B_components.py](../cs336_systems/benchmark_8B_components.py) · [flash_autotune.py](../cs336_systems/flash_autotune.py) |

The XL model has **3,406,809,600 parameters**: 32 layers, `d_model` 2,560, `d_ff` 10,240, 32 heads, and vocabulary size 10,000. Model dimensions, defaults, and timing procedures are defined in the benchmark implementations.

## Attention performance

**Hardware:** 1 × A100 80G.  
**Workload:** causal attention, batch 1, head dimension 128, BF16.  
**Metric:** forward-plus-backward latency.

| Sequence length | Explicit PyTorch (ms) | Custom Triton (ms) | Speedup |
| ---: | ---: | ---: | ---: |
| 1,024 | 0.363 | 0.213 | 1.70× |
| 2,048 | 0.408 | 0.271 | 1.50× |
| 4,096 | 0.790 | 0.539 | 1.46× |
| 8,192 | 2.876 | 1.334 | 2.16× |
| 16,384 | 9.988 | 3.780 | 2.64× |
| **32,768** | **40.825** | **13.080** | **3.12×** |

These measurements use the 32×64 tile configuration; the tiles were later retuned for the 8B shape (see [attention tile tuning](#attention-tile-tuning)). Speedup is reference latency divided by custom-kernel latency, calculated before rounding. The reference materializes attention scores and uses explicit matrix multiplication, masking, and softmax; it does not call optimized PyTorch SDPA. Inputs and masks are constructed outside the timed function. The [benchmark](../cs336_systems/flash_benchmarking.py) performs an explicit warmup and uses `triton.testing.do_bench`.

## Optimizer-state sharding

**Hardware:** 2 × A100 80G.  
**Workload:** XL model, local batch 2, context 512, FP32.

| Metric | Ordinary AdamW | Sharded AdamW | Change |
| --- | ---: | ---: | ---: |
| Optimizer-step peak memory, maximum rank (GB) | 54.879 | 41.457 | **−24.46%** |
| Mean iteration time across rank means (ms) | 1,717.416 | 1,747.099 | **+1.73%** |
| Post-update allocated memory, rank 0 (GB) | 54.572 | 41.150 | −13.422 GB |
| Post-update allocated memory, rank 1 (GB) | 54.572 | 40.740 | −13.832 GB |

The memory that stays allocated throughout training (parameters, gradients, and AdamW’s two moment buffers) can be calculated from the parameter count. For the 3.41-billion-parameter XL model in FP32, that is 13.6 GB each for parameters and gradients, and 27.3 GB for the moment buffers. Sharding the moment buffers across two ranks should therefore save 13.6 GB per rank; the measured savings are 13.4 and 13.8 GB, which average 13.6 GB. They differ slightly because each rank owns whole tensors, so the two shards are not exactly equal in size. See the [benchmark implementation](../cs336_systems/benchmark_sharded_optim.py) for the measurement procedure.

## Distributed data parallelism

**Hardware:** 2 or 4 × A100 80G.  
**Workload:** XL model, local batch 2, context 512, FP32.  
Each GPU measures its own average step time. The table reports the slowest GPU’s average. See the [DDP benchmark](../cs336_systems/benchmark_ddp.py).

| Variant | 2 GPUs: step time (ms) | 4 GPUs: step time (ms) |
| --- | ---: | ---: |
| Naive DDP | 1,895.532 | 1,798.515 |
| Flattened DDP | 1,902.281 | 1,806.564 |
| **Overlapped DDP** | **1,709.533** | **1,726.783** |
| **Overlap: reduction versus naive** | **9.81%** | **3.99%** |

## Fully sharded data parallelism

**Hardware:** 2 × A100 80G.  
**Workload:** XL model, local batch 2, context 512, FP16 compute with FP32 master shards.

| Metric | Result |
| --- | ---: |
| Rank 0 mean step time (ms) | 537.967 |
| Rank 1 mean step time (ms) | 538.030 |
| Maximum rank mean step time (ms) | **538.030** |

This run uses FP16 compute while the DDP runs use FP32, so the step times are not directly comparable.

## 8B training step on four A100 GPUs

**Hardware:** 4 × A100 80G.  
**Workload:** the 8.13-billion-parameter model (34 layers, `d_model` 4,096, `d_ff` 11,008, 32 heads, vocabulary 151,936), global batch 4, local batch 1, context 32,768.  
**Configuration:** BF16 compute, FP32 master shards, FSDP, fused AdamW, fused LM-head loss with its weight-gradient multiplication in TF32, and Triton attention with the tuned tiles below. Layers with `i % 16 == 15` keep their activations, so 32 layers are recomputed during backward.

### Step benchmark

| Metric | Result |
| --- | ---: |
| Mean step time, maximum over ranks (ms) | **22,451.916** |
| Global throughput (tokens/s) | **5,837.90** |
| Model FLOPs per step (PFLOP) | 9.489 |
| Executed FLOPs per step, including recomputation (PFLOP) | 14.271 |
| **MFU** / HFU (312 TFLOP/s dense BF16 peak) | **33.9%** / 50.9% |
| Baseline / peak allocated memory, maximum rank (GiB) | 23.266 / **56.730** |
| Peak reserved memory, maximum rank (GiB) | 69.174 |

Sources: [benchmark output](8B_a100/benchmark_8B_k16.txt) and [MFU calculation](8B_a100/mfu_calculation.txt), using [benchmark_8B.py](../cs336_systems/benchmark_8B.py) and [mfu.py](../cs336_systems/mfu.py). One warmup step and three measured steps.

### TF32 fix in the fused loss

The loss's backward pass accumulates the weight gradient with `grad_weight.addmm_` on FP32 copies of BF16 tensors. PyTorch runs FP32 matrix multiplications on CUDA cores unless TF32 is allowed. TF32 keeps 10 mantissa bits and BF16 has 7, so BF16 values are exact in TF32; [fused_lm_head_loss.py](../cs336_systems/fused_lm_head_loss.py) therefore allows TF32 for this call only, when the inputs are BF16.

| Metric | FP32 multiplication | TF32 multiplication |
| --- | ---: | ---: |
| Mean step time (ms) | 24,107.591 | **22,451.916 (−6.9%)** |
| MFU | 31.5% | **33.9%** |
| Weight-gradient kernel per step | ampere_sgemm_128x128_nt, 2,167.2 ms | Kernel2, 504.5 ms |
| LM head + loss backward per step (ms) | 2,628.1 | 971.5 |
| Peak allocated memory (GiB) | 56.730 | 56.730 |
| Final distributed mean loss | 12.720229 | 12.720284 |

Both the FP32 CUDA-core kernel and the TF32 kernel run 128 times per step, once per 256-token tile. The TF32 kernel reaches about 81 TFLOP/s, above the A100's 19.5 TFLOP/s FP32 CUDA-core peak, so it runs on tensor cores. The FP32 column comes from the runs before the fix: [benchmark](8B_a100/before_tf32_fix/benchmark_8B_k16.txt) (an earlier repeat measured 24,077.886 ms), [MFU calculation](8B_a100/before_tf32_fix/mfu_calculation.txt), [profile](8B_a100/before_tf32_fix/profile/8B-a100.txt), and [component benchmarks](8B_a100/before_tf32_fix/components/a100-components.txt).

### Nsight Systems profile

Two steady-state steps after two warmup steps, traced with [profile_8B.py](../cs336_systems/profile_8B.py) and summarized by [inspect_nsys_8B_step.py](../cs336_systems/inspect_nsys_8B_step.py). A profiled step takes 22,610 ms, 0.7% longer than the benchmark. Values are means over four ranks and two steps. Sources: [analysis output](8B_a100/profile/8B-a100.txt), [Nsight kernel summary](8B_a100/profile/8B-a100_cuda_gpu_kern_sum.csv), and [NVTX projection](8B_a100/profile/8B-a100_nvtx_gpu_proj_sum.csv).

| Phase | GPU time per step (ms) | Share |
| --- | ---: | ---: |
| Forward, including the fused loss | 5,715.3 | 25.3% |
| Backward | 16,852.0 | 74.5% |
| — of which activation recomputation | 4,172.7 | 18.5% |
| Gradient synchronization wait | 13.8 | 0.1% |
| Optimizer | 34.9 | 0.2% |

| Kernel category | GPU time per step (ms) | Share | Executed TFLOP/s | Share of peak |
| --- | ---: | ---: | ---: | ---: |
| Attention forward | 4,089.5 | 18.1% | 142.0 | 45.5% |
| Attention backward (dQ and dK/dV) | 6,334.4 | 28.0% | 165.2 | 53.0% |
| Projection and SwiGLU matrix multiplications | 6,324.2 | 28.0% | 281.0 | 90.1% |
| LM head + loss forward | 1,234.7 | 5.5% | 33.0 | 10.6% |
| LM head + loss backward | 971.5 | 4.3% | 125.9 | 40.4% |
| Other kernels (norms, RoPE, gating, casts, residual additions) | 3,375.8 | 14.9% | | |
| Optimizer | 34.6 | 0.2% | | |
| Memory copies and sets | 63.5 | 0.3% | | |
| NCCL all-gather and reduce-scatter | 1,304.9 | 5.8% | | |

Category times are summed kernel durations; NCCL kernels run on separate streams and overlap computation. Executed TFLOP/s counts the matrix multiplications each category actually performs, including recomputed attention scores and logits.

| Rank | Compute busy | Communication busy | Exposed communication | Idle |
| --- | ---: | ---: | ---: | ---: |
| GPU 0 | 99.6% | 4.1% | 0.3% | 0.1% |
| GPU 1 | 98.8% | 6.8% | 1.1% | 0.1% |
| GPU 2 | 99.1% | 6.0% | 0.8% | 0.1% |
| GPU 3 | 99.2% | 6.2% | 0.7% | 0.1% |
| **Mean** | **99.2%** | **5.8%** | **0.7%** | **0.1%** |

Exposed communication is time when NCCL kernels run with no other GPU work. It includes NCCL kernels waiting for slower ranks.

The GPUs are almost never waiting. They spend 99.2% of the step computing.  
About 88% of the communication is hidden. Of the 5.8% of the step when NCCL is running, only 0.7% is exposed.  
Idle time is negligible. At 0.1%, there are essentially no gaps from CPU launch overhead or synchronization.

### Component benchmarks

Each component runs alone on one A100 at one rank's shape, timed with `triton.testing.do_bench` ([benchmark_8B_components.py](../cs336_systems/benchmark_8B_components.py)). FLOP rates count model FLOPs; bandwidth uses each operation's minimum memory traffic. Sources: [output with the step budget](8B_a100/components/a100-components.txt) and [CSV](8B_a100/components/a100-components.csv).

| Component | Implementation | Forward (ms) | Forward + backward (ms) | Efficiency |
| --- | --- | ---: | ---: | --- |
| Causal attention, 32 heads | Triton (my implementation) | 61.509 | 245.239 | 143 / 108 TFLOP/s |
| | PyTorch SDPA, flash backend | 42.373 | 157.744 | 208 / 167 TFLOP/s |
| q/k/v/output projection | Linear | 3.910 | 11.802 | 90% of peak |
| SwiGLU w1/w3 and w2 | Linear | 10.541 / 10.302 | 31.157 / 31.287 | 90–92% of peak |
| RMSNorm | Assignment 1 module | 3.461 | 13.631 | 7.6% of bandwidth |
| RoPE | Assignment 1 module | 4.575 | 16.978 | 5.8% of bandwidth |
| SwiGLU gate | `silu(a) * b` | 3.293 | 10.529 | 32.2% of bandwidth |
| LM head + cross-entropy | Fused Triton (my implementation) | 1,233.191 | 2,188.992 | 33 / 56 TFLOP/s |
| | Materialized BF16 logits + `F.cross_entropy` | 167.917 | 493.789 | 243 / 248 TFLOP/s |
| Optimizer step, one rank's shard | Fused AdamW (my implementation) | | 34.926 | 79.9% of bandwidth |
| | Assignment 1 AdamW | | 163.922 | 17.0% of bandwidth |

Adding the component times for one step (34 layers forward + backward, 32 recomputed forwards, the loss, and the optimizer) gives 21,895 ms, against 22,452 ms measured. The remaining 557 ms (2.5%) covers communication, the embedding, residual additions, layout copies, and launch gaps; the profile attributes 0.8% of the step to exposed communication and idle time.

### Attention tile tuning

[flash_autotune.py](../cs336_systems/flash_autotune.py) searched 78 configurations at the 8B shape: 32 heads, 32,768 tokens, head dimension 128, BF16, causal. Full timings: [sweep output](8B_a100/flash_autotune.txt).

| Kernel | 32×64, 4 warps, 2 stages | Selected configuration | Selected | Speedup |
| --- | ---: | --- | ---: | ---: |
| Forward | 79.704 ms | Q=128, K=32, 4 warps, 2 stages | 61.440 ms | 1.30× |
| Backward dQ | 82.284 ms | Q=64, K=64, 4 warps, 2 stages | 72.005 ms | 1.14× |
| Backward dK/dV | 115.449 ms | Q=32, K=64, 4 warps, 3 stages | 108.850 ms | 1.06× |


### Interpretation

- **Communication is hidden.** FSDP prefetching overlaps all but 0.7% of the step's communication, and the GPUs are idle 0.1% of the time. The remaining step time is kernel execution.
- **Fused loss cost.** After the TF32 fix, the fused loss takes 2.2 s per step, against 0.5 s for a loss that computes the full logits. Most of the remaining gap is the forward kernel (1.23 s): it splits its work into only 128 pieces for the GPU's 108 SMs, which likely leaves most SMs idle during a second round of 20 pieces. Computing the full logits would need about 9.3 GiB more memory for the logits alone.
- **Optimizer.** Fused AdamW is 4.7× faster than the reference optimizer, but either takes under 1% of the step.
- **Scope.** These are synthetic training steps on random tokens, not training to convergence. The two-B200 runs keep every fourth layer's activations instead of every sixteenth; the latest run uses the A100-tuned tiles and the TF32 fix, and runs 1.29× faster than the first ([B200 comparison](section9/README.md#attention-tiles-and-tf32-loss-update)).


## Additional experiments and analysis

| Experiment | Configuration or implementation | Analysis |
| --- | --- | --- |
| Transformer timing | [benchmark.py](../cs336_systems/benchmark.py), with and without `--compiled` | — |
| Mixed precision | [benchmark_mixed_precision.py](../cs336_systems/benchmark_mixed_precision.py) | — |
| Attention baseline and compilation | [benchmark_attention.py](../cs336_systems/benchmark_attention.py), with and without `--compiled` | Separate from the custom Triton comparison above |
| Compute and memory profiling | [benchmark.py](../cs336_systems/benchmark.py) · [benchmark_mixed_precision.py](../cs336_systems/benchmark_mixed_precision.py) | — |

