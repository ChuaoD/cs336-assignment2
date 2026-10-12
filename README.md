# Memory-Efficient Distributed Transformer Training

**Custom Triton GPU kernels, FSDP, and activation checkpointing**

> **Part 2 of 2.** The model layers and optimizer used here come from my [from-scratch Transformer language model](https://github.com/ChuaoD/cs336-assignment1) (Part 1: BPE tokenizer, Transformer, AdamW, and training pipeline). This repository scales that code to an 8.1B-parameter model with custom GPU kernels and multi-GPU training.

This project implements a **memory-efficient distributed Transformer training system**, from custom GPU kernels to multi-GPU execution. It explores how to reduce attention and loss memory, distribute model and optimizer state, and overlap communication with computation while preserving numerical correctness.

I implemented Triton attention and fused training kernels, custom DDP and FSDP, optimizer-state sharding, and activation checkpointing, then integrated them into a complete training-step benchmark. My work spans kernel tiling and backward computation, asynchronous collective scheduling, checkpoint-aware weight prefetching, reference-based correctness tests, and analytical memory accounting checked against measurements. Experiments on **A100 80G** and **B200** GPUs connect these implementation choices to runtime and memory tradeoffs.

The project builds on **Stanford CS336 Assignment 2**. The contributions below describe my implementation, integration, and experimental work using the assignment's framework and established algorithms.

## My contributions

| Component | What I implemented | Engineering focus |
| --- | --- | --- |
| [FlashAttention](cs336_systems/flash_attention.py) | Tiled attention with online softmax, custom autograd, and Triton forward/backward kernels, including separate dQ and dK/dV passes | Reduce attention-matrix storage and memory traffic; exploit causal structure; tune tile sizes |
| [Distributed data parallelism](cs336_systems/ddp.py) | Naive, flattened, and asynchronous gradient synchronization variants | Measure communication overhead and overlap communication with backpropagation |
| [Optimizer-state sharding](cs336_systems/sharded_optimizer.py) | Distribute parameter ownership for optimizer updates and synchronize updated parameters | Quantify state-memory savings, ownership imbalance, and synchronization cost |
| [FSDP](cs336_systems/fsdp.py) | FP32 master shards, mixed-precision compute weights, asynchronous all-gather and reduce-scatter, and weight prefetching | Coordinate tensor lifetimes, gradient synchronization, and communication scheduling |
| [Activation checkpointing](cs336_systems/activation_checkpoint.py) | Custom autograd recomputation with explicit recomputation context, integrated with FSDP's weight-prefetch schedule | Preserve gradient behavior while trading saved activations for additional computation |
| [Fused LM head and cross-entropy](cs336_systems/fused_lm_head_loss.py) | Tiled projection/loss kernels and custom backward, avoiding the complete token-by-vocabulary logits tensor ([derivation](docs/fused_lm_head_loss_derivation.md)) | Control loss-phase memory while retaining the summed cross-entropy objective |
| [Fused AdamW](cs336_systems/fused_adamw.py) | A Triton kernel that updates parameters and moment buffers in place | Reduce optimizer intermediates and kernel launches; compare updates against the reference |
| [Integrated Transformer](cs336_systems/model_flash.py) | Connect custom attention, fused loss, and checkpointing to the distributed training path | Check interactions across kernels, autograd, and parameter sharding |

## Highlight: 8B training step

The integrated system runs a complete training step for an **8.13-billion-parameter Transformer with a 32,768-token context**, bringing together custom Triton attention, fused LM-head/cross-entropy, FSDP, activation checkpointing, and fused AdamW. Each GPU processes one sequence.

| Hardware | Full training-step time | Global throughput | MFU | Peak allocated memory per GPU |
| --- | ---: | ---: | ---: | ---: |
| **4 × A100 80GB** | **22.452 seconds** | **5,837.90 tokens/second** | **33.9%** | **56.730 GiB** |
| **2 × B200** | **7.530 seconds** | **8,702.83 tokens/second** | **14.0%** | **122.224 GiB** |

Both measurements include forward computation, loss, backward computation, gradient synchronization, and the optimizer update, with BF16 compute and FP32 master parameters. They benchmark synthetic training steps rather than a model trained to convergence. The runs recompute different numbers of layers during backward, as described below.

## 8B training on four A100 GPUs

The four-A100 run uses global batch 4, and 32 of the 34 layers are recomputed during backward.

MFU counts model FLOPs: six times the matrix-multiplication parameters per token, plus causal attention, with no credit for recomputation. It is measured against the A100's 312 TFLOP/s dense BF16 peak. Counting the work this implementation actually executes, including recomputation, gives **50.9% hardware FLOPs utilization**. See the [FLOP accounting](cs336_systems/mfu.py) and the [calculation](results/8B_a100/mfu_calculation.txt).

**Where the step goes.** The table below breaks the 22.5 s step into its main parts. It comes from an Nsight Systems trace of two steps of the real four-GPU run, taken after warmup ([analysis](results/8B_a100/profile/8B-a100.txt), [analysis script](cs336_systems/inspect_nsys_8B_step.py)). As a cross-check, each part was also benchmarked alone on one GPU at the same shapes ([component benchmarks](results/8B_a100/components/a100-components.txt)). The main kernels' times match the trace within 2%, and the parts add up to within 2.5% of the measured step time.

| Component | Time per step | Share | Efficiency |
| --- | ---: | ---: | --- |
| Attention (Triton forward, dQ, and dK/dV kernels) | 10.4 s | 46% | 142–165 TFLOP/s executed; PyTorch SDPA's flash backend is 1.55× faster in a single-GPU benchmark at this shape |
| Projection and SwiGLU matrix multiplications | 6.3 s | 28% | 281 TFLOP/s, 90% of peak |
| Norms, RoPE, gating, and other elementwise kernels | 3.4 s | 15% | 6–32% of HBM bandwidth when benchmarked alone |
| Fused LM head and cross-entropy | 2.2 s | 10% | 11% of peak forward, 40% backward |
| Exposed communication and idle time | 0.18 s | 0.8% | |

The GPUs are busy for **99.2%** of the step. NCCL all-gather and reduce-scatter kernels run for 5.8% of the step, of which only **0.7%** is not overlapped with computation, so FSDP's prefetching hides almost all communication. Shares are relative to the profiled step (22.610 s, including 0.7% tracing overhead).

**What the measurements show.**

- **Tile tuning.** A 78-configuration sweep at the training shape ([sweep](cs336_systems/flash_autotune.py), [results](results/8B_a100/flash_autotune.txt)) made the forward kernel 1.30× faster (79.7 → 61.4 ms), dQ 1.14× faster, and dK/dV 1.06× faster than the 32×64 tiles.
- **A slow multiplication in the fused loss, found and fixed.** The profile showed one multiplication in the loss's backward pass taking 2.2 s per step, 9% of the step. It used FP32 numbers, and by default FP32 runs on the GPU's regular cores instead of its much faster tensor cores. Its inputs start as BF16, which fit exactly into TF32, the tensor-core format for FP32 data, so allowing TF32 for that one call changes no input values. It now takes 0.5 s, and the step fell from **24.108 s to 22.452 s (6.9% faster)**, with unchanged memory and a final loss that moved by 0.00006.
- **The fused loss's forward kernel is still slow.** It splits the work into only 128 pieces, while the A100 has 108 processing units (SMs). After the first round of 108 pieces, the remaining 20 likely leave most of the GPU idle. That kernel takes 1.2 s. Computing the full logits instead is 4.4× faster for forward + backward when benchmarked alone, but needs about 9.3 GiB more memory for the logits alone.
- **Fused AdamW.** It updates a rank's 2.03-billion-parameter shard in 35 ms at 80% of HBM bandwidth, 4.7× faster than the reference optimizer (164 ms). However, either optimizer takes under 1% of the step.

**Configuration notes.** Layers with `i % 16 == 15` keep their activations ([model](cs336_systems/model_flash.py)). The measured peak memory falls within the 48–60 GiB estimated analytically before the run. The [step benchmark](cs336_systems/benchmark_8B.py) uses one warmup step and three measured steps. [Detailed tables](results/README.md#8b-training-step-on-four-a100-gpus).

## 8B training on two B200 GPUs

The two-B200 run uses the attention tiles tuned on A100 (above) and the TF32 fix in the fused loss. The Triton kernels have not been tuned for Blackwell.

| Change, applied in order | Step time | Faster than the previous row |
| --- | ---: | ---: |
| Attention tiles (32 × 64) | 9.743 s | |
| A100-tuned attention tiles | 8.229 s | 15.5% |
| TF32 for the fused loss's weight-gradient multiplication | **7.530 s** | 8.5% |

Together the two changes make the step **1.29× faster**. MFU rises from 10.8% to 14.0%. It stays lower than on A100 because the B200's dense BF16 peak (2,250 TFLOP/s) is 7.2× the A100's, while the kernels were tuned on A100. [B200 comparison](results/section9/README.md#attention-tiles-and-tf32-loss-update).

Configuration: global batch 2 and fused AdamW. Every fourth layer keeps its activations (`i % 4 == 3`, so 26 of 34 layers are recomputed); reproducing it requires changing the rule in [model_flash.py](cs336_systems/model_flash.py) from `i % 16 != 15 to i % 4 != 3`, in both places. [Full benchmark configuration and results](results/section9/README.md).

## Selected results

| Experiment | Hardware | Result | Detailed tables |
| --- | --- | --- | --- |
| **8B full training step** | 4 × A100 80G | **22.452 s/step; 5,838 tokens/s; 33.9% MFU**; 0.7% exposed communication | [A100 8B results](#8b-training-on-four-a100-gpus) |
| **Attention tile tuning** | 1 × A100 80G | Forward **1.30× faster** (79.7 → 61.4 ms) at the 8B shape; 143 TFLOP/s, about 70% of PyTorch SDPA's flash backend | [Tile sweep](results/8B_a100/flash_autotune.txt) |
| **Fused AdamW step** | 1 × A100 80G | **4.7× faster**: 163.9 → 34.9 ms for one rank's shard; 80% of HBM bandwidth | [Component benchmarks](results/8B_a100/components/a100-components.txt) |
| **8B full training step** | 2 × B200 | **7.530 s/step; 8,702.8 tokens/s; 14.0% MFU** | [B200 8B results](results/section9/README.md#attention-tiles-and-tf32-loss-update) |
| **Triton causal attention** | 1 × A100 80G | **3.12× speedup**: 40.825 → 13.080 ms for forward + backward | [Attention](results/README.md#attention-performance) |
| **Optimizer-state sharding** | 2 × A100 80G | **24.5% lower optimizer-step peak memory**, with **1.7% higher iteration time** | [Memory and runtime](results/README.md#optimizer-state-sharding) |
| **Overlapping gradient communication** | 2 × A100 80G | **9.8% lower step time**: 1,895.532 → 1,709.533 ms | [DDP](results/README.md#distributed-data-parallelism) |


## Design decisions worth inspecting

**Checkpointing changes the communication schedule.** FSDP gathers weights as layers execute, then restores local shards. Activation checkpointing executes a block again during backward. I added an explicit recomputation context and block-local execution tracking so that weight prefetching follows the recomputed forward pass without advancing the ordinary forward schedule. See [FSDP](cs336_systems/fsdp.py) and the [checkpointed distributed integration test](tests/test_fsdp.py).

**The output projection is also a memory bottleneck.** At local batch 1, context 32,768, and vocabulary 151,936, a complete BF16 logits tensor alone occupies **9.273 GiB per rank**. The fused loss computes the projection and normalization in tiles, saves log-sum-exp values, and recomputes values needed by backward. See the [loss implementation](cs336_systems/fused_lm_head_loss.py) and its [derivation and design](docs/fused_lm_head_loss_derivation.md).

## Correctness and validation

The test suite includes reference comparisons at kernel, model, and distributed-training levels:

- [Attention](tests/test_attention.py).
- [Fused loss](tests/test_fused_loss.py).
- [Fused AdamW](tests/test_fused_adamw.py).
- [Activation checkpointing](tests/test_activation_checkpoint.py).
- [Transformer integration](tests/test_model_flash.py).
- [FSDP integration](tests/test_fsdp.py).

The Triton attention kernels use `tl.dot`'s default precision, which runs FP32 inputs as TF32 on tensor cores. Two FP32 tests in [tests/test_model_flash.py](tests/test_model_flash.py) (`test_multihead_self_attention_backward` and `test_transformer_lm_backward`) exceed their tolerances (1e-3 and 5e-3) under TF32 and pass when the commented `input_precision="ieee"` lines in [flash_attention.py](cs336_systems/flash_attention.py) are enabled. Training uses BF16 inputs, which TF32 does not affect.


## Running the project

The GPU kernels and distributed benchmarks target **Linux, NVIDIA CUDA GPUs, Triton, and NCCL**. Use Python 3.12 or 3.13 and `uv`. Two visible GPUs are required for the distributed integration tests; reproducing the B200 result requires two B200s.

### Dependency layout

The current environment uses my Assignment 1 implementation as an editable sibling dependency:

```text
parent/
  cs336-assignment1/
    pyproject.toml
    cs336_basics/
  assignment2-systems/
    pyproject.toml
    cs336_systems/
```

Both [pyproject.toml](pyproject.toml) and the [Modal launcher](cs336_systems/modal_b200_benchmark.py) refer to this sibling package, which is my [Assignment 1 repository](https://github.com/ChuaoD/cs336-assignment1). Clone it next to this repository before running `uv sync`.

From this repository, with that dependency available:

```bash
uv sync --frozen

# Small CPU checkpointing checks
uv run pytest tests/test_activation_checkpoint.py -q

# Kernel and model checks on a CUDA machine
uv run pytest tests/test_attention.py tests/test_fused_loss.py tests/test_fused_adamw.py tests/test_model_flash.py -q

# Distributed integration on a two-GPU machine; tests launch their own workers
uv run pytest tests/test_fsdp.py -q
```

### Benchmarks

```bash
# Attention sweep on one GPU; writes CSVs and plots under benchmark_flash/
uv run python -m cs336_systems.flash_benchmarking

# Section 9 configuration on two visible B200 GPUs; launches its own workers
uv run python -m cs336_systems.benchmark_8B_b200 \
  --warmup-ms 10000 --rep-ms 30000 --activation-checkpointing \
  --optimizer fused-adamw
```

The four-A100 results use these commands:

```bash
# 8B training step on four visible A100 80GB GPUs
uv run python -m cs336_systems.benchmark_8B --world-size 4 --global-batch-size 4

# MFU and HFU for a measured step time
uv run python -m cs336_systems.mfu --gpu a100 --num-gpus 4 --global-batch-size 4 \
  --step-ms 22452 --recomputed-layers 32

# Component benchmarks on one A100, compared with the measured step time
uv run python -m cs336_systems.benchmark_8B_components --gpu a100 --world-size 4 \
  --step-ms 22452 --recomputed-layers 32

# Nsight Systems profile of two steady-state steps on four A100s, then the summary
NSYS_NVTX_PROFILER_REGISTER_ONLY=0 nsys profile \
  --trace=cuda,nvtx --sample=none \
  --capture-range=nvtx --nvtx-capture=profile --capture-range-end=none \
  --wait=all -o 8B-a100 \
  uv run python -m cs336_systems.profile_8B --world-size 4
nsys export --type=sqlite -o 8B-a100.sqlite 8B-a100.nsys-rep
uv run python -m cs336_systems.inspect_nsys_8B_step 8B-a100.sqlite --gpu a100
```

Use `--optimizer adamw` with the same settings to run with the ordinary Assignment 1 AdamW instead. The default is `fused-adamw` in both 8B benchmark runners, including runs launched through Modal. Validate the fused optimizer with `uv run pytest tests/test_fused_adamw.py tests/test_fsdp.py -k fused_adamw -q` on a two-GPU CUDA machine before the full benchmark.

The optional [Modal launcher](cs336_systems/modal_b200_benchmark.py) requests two B200s and pins its image dependencies. With a configured Modal account and the sibling dependency available, it can be invoked with `uv run --with modal modal run cs336_systems/modal_b200_benchmark.py --profile section9`.

## Scope and next comparisons

The project demonstrates implementation correctness, memory and communication scheduling, and measured performance tradeoffs. The attention comparison measures the benefit of the custom FlashAttention implementation over explicit PyTorch attention. The distributed comparisons evaluate this project's synchronization and sharding choices.

The A100 profile and component benchmarks identify where the 8B step's time goes. The largest remaining optimization opportunities are, in order of estimated savings per step:
- closing the gap between the Triton attention kernels and PyTorch SDPA (about 3.6 s);
- fusing the norm and RoPE kernels, which run at 6–8% of HBM bandwidth (about 2.5 s);
- splitting the fused loss's forward work into more pieces, or computing the loss in chunks with cuBLAS (about 1.7 s).

These estimates come from component benchmarks and have not been measured end to end. PyTorch SDPA and PyTorch FSDP are external reference points, not the project's success criterion. No claim of outperforming those implementations is made.

The custom FSDP targets the project's Linear/Embedding modules and traced execution order; each optimizer step uses one forward and backward pass. It does not support gradient accumulation (summing gradients over several smaller batches before one update), so the global batch is the number of GPUs times the sequences per GPU.

## Attribution

The assignment specification, starter scaffolding, reference material, and portions of the test infrastructure originate from Stanford CS336. FlashAttention and FSDP are established algorithms; this project demonstrates my implementation and integration of those techniques. Several advanced components implement optional extensions suggested by the assignment. See the [assignment PDF](cs336_assignment2_systems.pdf) and retained [license](LICENSE).
