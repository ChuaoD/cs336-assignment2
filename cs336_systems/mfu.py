"""Analytical FLOP accounting and utilization for the Transformer.

Model FLOPs: one forward pass plus a backward pass costing twice the forward pass, with no credit for recomputation.

Implementation FLOPs count the matrix multiplications this project actually
executes. They add the forward passes recomputed by activation checkpointing,
the logit recomputation in the fused LM-head backward, and the attention
scores recomputed by both the dQ pass and the dK/dV pass of the FlashAttention
backward. They define hardware FLOPs utilization (HFU).

Only matrix multiplications are counted. Norms, RoPE, SwiGLU gating, softmax
exponentials, and the optimizer update are memory-bound and together account
for under one percent of the FLOPs at the 8B shape.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, fields


@dataclass(frozen=True)
class GPUSpec:
    name: str
    peak_bf16_flops: float  # dense tensor-core FLOP/s, no structured sparsity
    memory_bandwidth: float  # HBM bytes/s


GPU_SPECS = {
    "a100": GPUSpec("NVIDIA A100 80GB SXM", 312e12, 2.039e12),
    "b200": GPUSpec("NVIDIA B200", 2.25e15, 8.0e12),
}


@dataclass(frozen=True)
class ModelShape:
    """Dimensions of TransformerLM in cs336_systems.model_flash."""

    vocab_size: int = 151_936
    context_length: int = 32_768
    d_model: int = 4_096
    d_ff: int = 11_008
    num_layers: int = 34
    num_heads: int = 32


def parameter_shapes(shape: ModelShape) -> list[tuple[int, ...]]:
    """Return every parameter shape of TransformerLM in named_parameters order.

    Linear and Embedding weights are the two-dimensional tensors that FSDP
    shards. RMSNorm weights are one-dimensional and stay replicated.
    """
    d_model, d_ff = shape.d_model, shape.d_ff
    block = [
        (d_model,),  # ln1
        (d_model,),  # ln2
        (d_ff, d_model),  # ffn.w1
        (d_model, d_ff),  # ffn.w2
        (d_ff, d_model),  # ffn.w3
        (d_model, d_model),  # attn.q_proj
        (d_model, d_model),  # attn.k_proj
        (d_model, d_model),  # attn.v_proj
        (d_model, d_model),  # attn.output_proj
    ]
    return (
        [(shape.vocab_size, d_model)]  # token_embeddings
        + block * shape.num_layers
        + [(d_model,), (shape.vocab_size, d_model)]  # ln_final, lm_head
    )


def parameter_count(shape: ModelShape) -> int:
    return sum(math.prod(s) for s in parameter_shapes(shape))


def checkpointed_layer_count(num_layers: int) -> int:
    """Mirror TransformerLM.checkpoint_blocks: every layer except each sixteenth."""
    return sum(1 for i in range(num_layers) if i % 16 != 15)


def block_linear_forward_flops(shape: ModelShape) -> int:
    """Forward FLOPs of one block's projections and SwiGLU, one sequence."""
    d_model, d_ff = shape.d_model, shape.d_ff
    weights = 4 * d_model * d_model + 3 * d_model * d_ff
    return 2 * shape.context_length * weights


def causal_attention_matmul_flops(shape: ModelShape) -> int:
    """FLOPs of one causal [T, T] x [T, d] product over all heads, one sequence.

    Without a mask, QK^T or PV costs 2 * T^2 * d_model summed over heads.
    The causal kernels skip tiles above the diagonal, which removes half of
    that work; the partially masked diagonal tiles are ignored.
    """
    return shape.context_length**2 * shape.d_model


def lm_head_forward_flops(shape: ModelShape) -> int:
    """Forward FLOPs of the output projection to vocabulary logits, one sequence."""
    return 2 * shape.context_length * shape.d_model * shape.vocab_size


@dataclass(frozen=True)
class FlopBreakdown:
    """Training-step FLOPs for one sequence, grouped by the kernels that run them."""

    block_matmuls: float
    attention_forward: float
    attention_backward: float
    lm_head_forward: float
    lm_head_backward: float

    @property
    def total(self) -> float:
        return sum(getattr(self, field.name) for field in fields(self))

    def scaled(self, factor: float) -> FlopBreakdown:
        return FlopBreakdown(
            **{field.name: getattr(self, field.name) * factor for field in fields(self)}
        )


def model_flops(shape: ModelShape) -> FlopBreakdown:
    """Model FLOPs: forward once, backward at twice the forward cost."""
    linear = block_linear_forward_flops(shape)
    attention = causal_attention_matmul_flops(shape)
    lm_head = lm_head_forward_flops(shape)
    layers = shape.num_layers
    return FlopBreakdown(
        block_matmuls=3 * layers * linear,
        attention_forward=layers * 2 * attention,  # QK^T, PV
        attention_backward=layers * 4 * attention,  # dV, dP, dQ, dK
        lm_head_forward=lm_head,
        lm_head_backward=2 * lm_head,  # d hidden, d weight
    )


def implementation_flops(shape: ModelShape, recomputed_layers: int) -> FlopBreakdown:
    """FLOPs executed by this project's kernels with checkpointed layers recomputed.

    FlashAttention backward runs two kernels. The dQ pass computes QK^T, dO V^T,
    and dS K; the dK/dV pass computes QK^T, P^T dO, dO V^T, and dS^T Q. The
    fused LM-head backward recomputes the logits before forming both gradients.
    """
    if not 0 <= recomputed_layers <= shape.num_layers:
        raise ValueError("recomputed_layers must lie between zero and num_layers")
    linear = block_linear_forward_flops(shape)
    attention = causal_attention_matmul_flops(shape)
    lm_head = lm_head_forward_flops(shape)
    layers = shape.num_layers
    return FlopBreakdown(
        block_matmuls=(3 * layers + recomputed_layers) * linear,
        attention_forward=(layers + recomputed_layers) * 2 * attention,
        attention_backward=layers * (3 + 4) * attention,
        lm_head_forward=lm_head,
        lm_head_backward=3 * lm_head,  # logits, d hidden, d weight
    )


def utilization(
    flops_per_step: float,
    step_seconds: float,
    num_gpus: int,
    peak_flops: float,
) -> float:
    """Fraction of the aggregate peak FLOP rate achieved by a training step."""
    return flops_per_step / (step_seconds * num_gpus * peak_flops)


def _print_breakdown(model: FlopBreakdown, implementation: FlopBreakdown) -> None:
    print(f"{'component':<22}{'model PFLOP':>14}{'impl. PFLOP':>14}")
    for field in fields(FlopBreakdown):
        print(
            f"{field.name:<22}"
            f"{getattr(model, field.name) / 1e15:>14.3f}"
            f"{getattr(implementation, field.name) / 1e15:>14.3f}"
        )
    print(f"{'total':<22}{model.total / 1e15:>14.3f}{implementation.total / 1e15:>14.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FLOPs per training step, and MFU/HFU when a step time is given."
    )
    parser.add_argument("--gpu", choices=sorted(GPU_SPECS), default="a100")
    parser.add_argument(
        "--peak-tflops",
        type=float,
        default=None,
        help="override the dense BF16 peak of --gpu, in TFLOP/s per GPU",
    )
    parser.add_argument("--num-gpus", type=int, required=True)
    parser.add_argument("--global-batch-size", type=int, required=True)
    parser.add_argument("--step-ms", type=float, default=None)
    parser.add_argument("--context-length", type=int, default=32_768)
    parser.add_argument("--vocab-size", type=int, default=151_936)
    parser.add_argument("--d-model", type=int, default=4_096)
    parser.add_argument("--d-ff", type=int, default=11_008)
    parser.add_argument("--num-layers", type=int, default=34)
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument(
        "--recomputed-layers",
        type=int,
        default=None,
        help="layers recomputed by activation checkpointing "
        "(default: TransformerLM's rule, every layer except each sixteenth)",
    )
    args = parser.parse_args()

    if args.num_gpus <= 0:
        raise ValueError("num_gpus must be positive")
    if args.global_batch_size <= 0:
        raise ValueError("global_batch_size must be positive")
    if args.step_ms is not None and args.step_ms <= 0:
        raise ValueError("step_ms must be positive")

    shape = ModelShape(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        d_ff=args.d_ff,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
    )
    recomputed_layers = (
        checkpointed_layer_count(shape.num_layers)
        if args.recomputed_layers is None
        else args.recomputed_layers
    )
    gpu = GPU_SPECS[args.gpu]
    peak_flops = gpu.peak_bf16_flops if args.peak_tflops is None else args.peak_tflops * 1e12

    # Every sequence in the global batch costs the same FLOPs.
    model = model_flops(shape).scaled(args.global_batch_size)
    implementation = implementation_flops(shape, recomputed_layers).scaled(
        args.global_batch_size
    )

    print(f"gpu={gpu.name}")
    print(f"peak_bf16_tflops_per_gpu={peak_flops / 1e12:.1f}")
    print(f"num_gpus={args.num_gpus}")
    print(f"global_batch_size={args.global_batch_size}")
    print(f"context_length={shape.context_length}")
    print(f"num_layers={shape.num_layers}")
    print(f"recomputed_layers={recomputed_layers}")
    print(f"parameters={parameter_count(shape)}")
    print(f"global_tokens_per_step={args.global_batch_size * shape.context_length}")
    print()
    _print_breakdown(model, implementation)

    if args.step_ms is None:
        return

    step_seconds = args.step_ms / 1_000
    global_tokens = args.global_batch_size * shape.context_length
    print()
    print(f"step_ms={args.step_ms:.3f}")
    print(f"global_tokens_per_second={global_tokens / step_seconds:.2f}")
    print(
        "model_tflops_per_gpu="
        f"{model.total / step_seconds / args.num_gpus / 1e12:.1f}"
    )
    print(
        "implementation_tflops_per_gpu="
        f"{implementation.total / step_seconds / args.num_gpus / 1e12:.1f}"
    )
    print(f"mfu={utilization(model.total, step_seconds, args.num_gpus, peak_flops):.4f}")
    print(
        "hfu="
        f"{utilization(implementation.total, step_seconds, args.num_gpus, peak_flops):.4f}"
    )


if __name__ == "__main__":
    main()
