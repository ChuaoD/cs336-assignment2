# 8B Transformer training on two B200 GPUs

## Attention tiles and TF32 loss update

Two changes were applied one at a time. First, the attention tiles selected by the [A100 tile sweep](../8B_a100/flash_autotune.txt): forward Q=128, K=32; dQ Q=64, K=64; dK/dV Q=32, K=64 with three pipeline stages. Second, the fused loss's weight-gradient multiplication runs in TF32 instead of FP32 (see the [A100 results](../README.md#tf32-fix-in-the-fused-loss)). Everything else is kept the same: the runner and timing procedure, checkpointing (every fourth layer keeps its activations, so 26 of 34 layers are recomputed), fused AdamW, and the model.

| Metric | Previous tiles (32×64) | A100-tuned tiles | + TF32 loss |
| --- | ---: | ---: | ---: |
| **Mean full-step time (ms)** | **9,743.123** | **8,229.278 (−15.5%)** | **7,530.422 (−8.5%)** |
| **Global throughput (tokens/s)** | **6,726.39** | **7,963.76** | **8,702.83** |
| MFU / HFU | 10.8% / 15.7% | 12.8% / 18.6% | **14.0%** / 20.3% |
| Baseline allocated memory (GiB/GPU) | 61.344 | 61.346 | 61.346 |
| Peak allocated memory (GiB/GPU) | 122.220 | 122.224 | 122.224 |
| Peak reserved memory (GiB/GPU) | 141.795 | 141.795 | 141.795 |
| Final distributed mean loss | 11.274551 | 11.203506 | 11.203541 |

Percentages compare each run with the column to its left. Overall, the step is 1.29× faster and throughput is 29.4% higher.

**Finding:** the tiles chosen on A100 speed up the full B200 step by 15.5%, and the TF32 fix by a further 8.5%. The tiles have not been retuned on B200, where a separate sweep could select different configurations.

MFU uses the B200's 2,250 TFLOP/s dense BF16 peak and the [FLOP accounting](../../cs336_systems/mfu.py). Sources: [latest run](../8B_b200/benchmark_8B_k4.txt) and [MFU](../8B_b200/mfu_calculation.txt); [tuned-tiles-only run](../8B_b200/before_tf32_fix/benchmark_8B_k4.txt) and [MFU](../8B_b200/before_tf32_fix/mfu_calculation.txt); [previous-tiles run](../section9/b200-section9-fused-adamw_32_64.txt).

The final losses differ by 0.071 between the first two runs. Different tile shapes change the order of floating-point accumulation, and the difference compounds over the run's ten optimizer steps on random tokens. The TF32 fix moves the loss by only 0.00004, consistent with its inputs being exactly representable. These synthetic losses are not a correctness check; the kernels are validated against reference implementations by the test suite.


## Model and workload

| Setting | Value |
| --- | ---: |
| Hardware | **2 × NVIDIA B200** |
| Distributed ranks | 2 |
| Parameters | 8,125,698,048 |
| Global batch / local batch per GPU | 2 / 1 |
| Context length | 32,768 |
| Global tokens per optimizer step | 65,536 |
| Vocabulary size | 151,936 |
| `d_model` / `d_ff` | 4,096 / 11,008 |
| Transformer layers / attention heads | 34 / 32 |
| Compute precision | BF16 |
| Master parameters and optimizer states | FP32 |
| Causal attention | Enabled |
| Activation checkpointing | Enabled; 26 of 34 layers recomputed (`i % 4 != 3`) |
| Attention tiles | Previous 32×64, or A100-tuned (see above) |
| Fused-loss weight-gradient multiplication | FP32, or TF32 (latest run) |

The workload corresponds to Section 9 of the [assignment specification](../../cs336_assignment2_systems.pdf).

## Implementation and interpretation

All runs use custom Triton attention, fused LM-head/cross-entropy, FSDP, and activation checkpointing. FSDP averages gradients, so the local summed loss is multiplied by world size before backward to recover the global summed-loss objective. The [benchmark runner](../../cs336_systems/benchmark_8B_b200.py) defaults to `FusedAdamW`; `--optimizer adamw` selects the ordinary Assignment 1 optimizer.

This is a synthetic training-step benchmark. Similar final losses do not replace parameter, gradient, and optimizer-state correctness tests or establish language-model convergence. The exact runtime versions and physical GPU allocation were not captured for comparison across runs; the [launcher](../../cs336_systems/modal_b200_benchmark.py) specifies PyTorch 2.11.0 and Triton 3.6.0 as its environment configuration.
