# Section 2.1.4 - Nsight Systems profiling

## Measurement notes

- Hardware: NVIDIA A100; batch size 4; FP32.
- Each trace captures one measured step after five warmup steps.
- Forward wall time is the NVTX `forward` range duration and includes the explicit CUDA synchronization at the end of the range.
- Kernel percentages use cumulative GPU kernel duration from `cuda_gpu_kern_sum`.
- A kernel was classified as matrix multiplication when its name contained `gemm` or `gemv`.
- Attention timings sum the actual CUDA kernel durations attributed to the annotated NVTX ranges. The matrix-multiplication column includes only GEMM/GEMV kernels in `computing attention scores` and `final matmul`; the softmax column includes all six kernels in `computing softmax`.

## (a) Forward-pass time

| Model | Context length | Nsight forward time (ms) | Python time (ms) | Nsight difference |
|---|---:|---:|---:|---:|
| Small | 256 | 51.947 | - | - |
| Small | 512 | 63.584 | 49.911 | +27.4% |
| Small | 1024 | 118.533 | - | - |
| Medium | 256 | 115.471 | - | - |
| Medium | 512 | 168.157 | 145.280 | +15.7% |
| Medium | 1024 | 332.456 | - | - |

**Ready-to-submit response (2 sentences).** Nsight measured forward-pass times of 51.95, 63.58, and 118.53 ms for the small model and 115.47, 168.16, and 332.46 ms for the medium model at context lengths 256, 512, and 1024, respectively. At context length 512 these are 27.4% and 15.7% above the ordinary Python measurements of 49.91 and 145.28 ms, so they have the same scale and trend but Nsight is slower because tracing introduces instrumentation overhead.

## (b) Dominant CUDA kernel

| Model | Context | Forward top kernel | Forward calls | Forward GPU share | Full-step top kernel | Same kernel? |
|---|---:|---|---:|---:|---|:---:|
| Small | 256 | `ampere_sgemm_32x128_tn` | 60 | 36.8% | `ampere_sgemm_64x32_sliced1x4_nt` | No |
| Small | 512 | `ampere_sgemm_128x32_tn` | 24 | 31.4% | `ampere_sgemm_64x32_sliced1x4_nt` | No |
| Small | 1024 | `ampere_sgemm_128x64_tn` | 25 | 29.0% | `ampere_sgemm_64x32_sliced1x4_nt` | No |
| Medium | 256 | `ampere_sgemm_128x64_tn` | 169 | 78.2% | `ampere_sgemm_128x64_tn` | Yes |
| Medium | 512 | `ampere_sgemm_128x32_tn` | 48 | 35.4% | `ampere_sgemm_128x64_nn` | No |
| Medium | 1024 | `ampere_sgemm_128x64_tn` | 145 | 46.1% | `ampere_sgemm_128x64_tn` | Yes |

The supplied second trace is a complete training step rather than forward-plus-backward only. Removing the optimizer-attributed kernel durations does not change the identity of the dominant full-step kernel shown above.

**Ready-to-submit response (2 sentences).** The most expensive forward kernel depends on shape: the small model uses `ampere_sgemm_32x128_tn` (60 calls), `ampere_sgemm_128x32_tn` (24 calls), and `ampere_sgemm_128x64_tn` (25 calls) at contexts 256, 512, and 1024, while the medium model uses `ampere_sgemm_128x64_tn` (169 calls), `ampere_sgemm_128x32_tn` (48 calls), and `ampere_sgemm_128x64_tn` (145 calls). In the training-step profiles, the dominant kernel changes for all three small-model contexts and for medium/512, but remains `ampere_sgemm_128x64_tn` for medium/256 and medium/1024.

## (c) Other significant forward kernels

| Model | Context | Matmul share | All non-matmul kernels |
|---|---:|---:|---:|
| Small | 256 | 80.7% | 19.3% |
| Small | 512 | 76.0% | 24.0% |
| Small | 1024 | 69.9% | 30.1% |
| Medium | 256 | 83.9% | 16.1% |
| Medium | 512 | 79.9% | 20.1% |
| Medium | 1024 | 73.3% | 26.7% |

**Ready-to-submit response (2 sentences).** Besides GEMMs, non-trivial forward time is spent in direct-copy kernels, elementwise multiplication/division/addition, attention masking, exponentiation, and max/sum reductions used by softmax. Together the non-matmul kernels account for 16.1%-30.1% of cumulative forward GPU kernel time, and their share generally grows with context length.

## (d) Forward inference versus a complete training step

| Model | Context | Forward matmul share | Full-step matmul share | Change |
|---|---:|---:|---:|---:|
| Small | 256 | 80.7% | 65.2% | -15.5 pp |
| Small | 512 | 76.0% | 66.9% | -9.1 pp |
| Small | 1024 | 69.9% | 64.3% | -5.6 pp |
| Medium | 256 | 83.9% | 68.0% | -15.9 pp |
| Medium | 512 | 79.9% | 70.4% | -9.5 pp |
| Medium | 1024 | 73.3% | 67.5% | -5.8 pp |

**Ready-to-submit response (2 sentences).** Matrix multiplication accounts for 69.9%-83.9% of cumulative GPU kernel time in forward-only inference, but only 64.3%-70.4% in a complete training step, a reduction of 5.6-15.9 percentage points across these profiles. Backpropagation and AdamW introduce many elementwise add, multiply, divide, and copy kernels, increasing the combined non-matmul share to 29.6%-35.7%.

## (e) Attention softmax versus matrix multiplication

All times below are cumulative over every attention layer in one forward pass.

| Model | Context | Two attention matmuls (ms) | Softmax (ms) | Matmul / softmax runtime |
|---|---:|---:|---:|---:|
| Small | 256 | 1.027 | 0.851 | 1.21x |
| Small | 512 | 3.702 | 4.275 | 0.87x |
| Small | 1024 | 13.802 | 15.520 | 0.89x |
| Medium | 256 | 3.542 | 2.313 | 1.53x |
| Medium | 512 | 10.340 | 11.091 | 0.93x |
| Medium | 1024 | 35.097 | 41.000 | 0.86x |

Let `B` be batch size, `H` the number of heads, `N` the context length, and `d_h` the head dimension. Here `B = 4`; the small model has `H = 12`, the medium model has `H = 16`, and both have `d_h = d_model / H = 64`.

The score multiplication has shapes `[B, H, N, d_h] @ [B, H, d_h, N]`. Using the standard convention that a multiply-add costs two FLOPs,

`QK^T FLOPs = 2 * B * H * N^2 * d_h`.

The multiplication of attention probabilities by values has the same cost,

`PV FLOPs = 2 * B * H * N^2 * d_h`.

Therefore, the two matrix multiplications together require

`matmul FLOPs = 4 * B * H * N^2 * d_h = 4 * B * N^2 * d_model`.

For each of the `B * H * N` softmax rows of length `N`, a numerically stable naive softmax performs a maximum reduction (`N - 1` comparisons), subtracts the maximum (`N` subtractions), computes exponentials (`N` exponentials), computes their sum (`N - 1` additions), and normalizes (`N` divisions). Counting each as one high-level operation gives `5N - 2` operations per row, so the total is

`softmax operations = B * H * N * (5N - 2) ~= 5 * B * H * N^2`.

| Model | Context | `QK^T` (GFLOPs) | `PV` (GFLOPs) | Both matmuls (GFLOPs) | Softmax (GFLOPs) | FLOP ratio |
|---|---:|---:|---:|---:|---:|---:|
| Small | 256 | 0.403 | 0.403 | 0.805 | 0.0157 | 51.2x |
| Small | 512 | 1.611 | 1.611 | 3.221 | 0.0629 | 51.2x |
| Small | 1024 | 6.442 | 6.442 | 12.885 | 0.2517 | 51.2x |
| Medium | 256 | 0.537 | 0.537 | 1.074 | 0.0210 | 51.2x |
| Medium | 512 | 2.147 | 2.147 | 4.295 | 0.0839 | 51.2x |
| Medium | 1024 | 8.590 | 8.590 | 17.180 | 0.3355 | 51.2x |

For example, for the small model at context 256:

`QK^T = 2 * 4 * 12 * 256^2 * 64 = 402,653,184 FLOPs`,

`PV = 402,653,184 FLOPs`, and

`softmax ~= 5 * 4 * 12 * 256^2 = 15,728,640 FLOPs`.

Thus, `(402,653,184 + 402,653,184) / 15,728,640 = 51.2x`. The absolute FLOP counts increase quadratically with context length and are larger for the medium model because it has more heads, but the ratio remains 51.2x because both models have the same head dimension. This is a high-level operation count: treating exponentials differently or counting the exact `N - 1` reduction operations changes the estimate slightly but not the conclusion.

**Ready-to-submit response (2 sentences).** Across the six traces, the two attention matrix multiplications take 0.86x-1.53x as much CUDA kernel time as softmax; at contexts 512 and 1024, softmax is actually slightly slower. This is dramatically smaller than the roughly 51x difference in high-level FLOPs for head dimension 64, because GEMMs use highly optimized compute-dense kernels whereas the naive softmax launches several comparatively memory-bound elementwise and reduction kernels.

## Data-provenance caveat

All 12 `.nsys-rep` profiles were generated successfully, and the supplied archive contains all 48 expected CSV tables. However, ten `nsys stats` invocations warned that their existing SQLite export was older than the newly generated `.nsys-rep`; therefore, rerun the statistics export with `--force-export=true` if strict correspondence to the latest traces is required. The report above analyzes the supplied CSV archive exactly as provided.
