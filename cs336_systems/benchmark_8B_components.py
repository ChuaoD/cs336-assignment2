from __future__ import annotations

import argparse
import csv
import math
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
import triton
from einops import einsum, rearrange
from torch.nn.attention import SDPBackend, sdpa_kernel

from cs336_basics.model import Linear, RMSNorm, RotaryPositionalEmbedding
from cs336_basics.optimizer import AdamW
from cs336_basics.utils import silu
from cs336_systems.flash_attention import FlashAttentionTriton
from cs336_systems.fused_adamw import FusedAdamW
from cs336_systems.fused_lm_head_loss import FusedLMHeadCrossEntropyFunction
from cs336_systems.mfu import (
    GPU_SPECS,
    GPUSpec,
    ModelShape,
    causal_attention_matmul_flops,
    checkpointed_layer_count,
    lm_head_forward_flops,
    parameter_shapes,
)

DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16  # FSDP compute dtype in the 8B runners

FORWARD = "forward"
FORWARD_BACKWARD = "forward+backward"
STEP = "step"

# Component names shared by the measurements and the step budget.
ATTENTION = "attention core"
QKVO_PROJECTION = "q/k/v/output projection"
SWIGLU_UP = "swiglu w1/w3"
SWIGLU_DOWN = "swiglu w2"
RMSNORM = "rmsnorm"
ROPE = "rope"
SWIGLU_GATE = "swiglu gate"
LM_HEAD_LOSS = "lm head + loss"
OPTIMIZER = "optimizer"

FUSED_ADAMW = "fused adamw (project)"
ADAMW = "adamw (assignment 1)"

# How many times each component runs in one Transformer block.
BLOCK_COMPONENT_COUNTS = {
    ATTENTION: 1,
    QKVO_PROJECTION: 4,
    SWIGLU_UP: 2,
    SWIGLU_DOWN: 1,
    RMSNORM: 2,
    ROPE: 2,  # queries and keys
    SWIGLU_GATE: 1,
}


@dataclass(frozen=True)
class Measurement:
    component: str
    implementation: str
    mode: str
    ms: float | None  # None when the component did not fit in memory
    flops: float | None = None
    min_bytes: float | None = None
    reference: bool = False  # an external baseline the 8B runners do not execute


Timer = Callable[[Callable[[], object]], float | None]


def make_timer(warmup_ms: int, rep_ms: int) -> Timer:
    """Return a timer that reports mean milliseconds, or None after an OOM."""

    def time_ms(fn: Callable[[], object]) -> float | None:
        try:
            fn()  # explicit warm up: Triton compilation and allocator growth
            torch.cuda.synchronize()
            return triton.testing.do_bench(fn, warmup=warmup_ms, rep=rep_ms)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return None

    return time_ms


def forward_backward(
    forward: Callable[[], torch.Tensor],
    inputs: tuple[torch.Tensor, ...],
    grad_output: torch.Tensor | None,
) -> Callable[[], tuple[torch.Tensor, ...]]:
    """Run forward, then compute gradients for inputs and parameters.

    torch.autograd.grad returns the gradients instead of accumulating them into
    .grad, so repeated calls do not add accumulation kernels to the timing.
    A scalar output may pass grad_output=None.
    """

    def fn() -> tuple[torch.Tensor, ...]:
        output = forward()
        return torch.autograd.grad(output, inputs, grad_outputs=grad_output)

    return fn


def tensor_bytes(*tensors: torch.Tensor) -> int:
    return sum(t.numel() * t.element_size() for t in tensors)


def measure_forward_and_backward(
    component: str,
    implementation: str,
    forward: Callable[[], torch.Tensor],
    inputs: tuple[torch.Tensor, ...],
    grad_output: torch.Tensor | None,
    timer: Timer,
    forward_flops: float | None = None,
    forward_bytes: float | None = None,
    reference: bool = False,
) -> list[Measurement]:
    backward_flops = None if forward_flops is None else 3 * forward_flops
    return [
        Measurement(
            component,
            implementation,
            FORWARD,
            timer(forward),
            forward_flops,
            forward_bytes,
            reference,
        ),
        Measurement(
            component,
            implementation,
            FORWARD_BACKWARD,
            timer(forward_backward(forward, inputs, grad_output)),
            backward_flops,
            reference=reference,
        ),
    ]


def benchmark_attention(shape: ModelShape, timer: Timer) -> list[Measurement]:
    """Causal attention over all heads, against PyTorch's FlashAttention-2 backend."""
    head_dim = shape.d_model // shape.num_heads
    # The project kernel takes (batch * heads, sequence, head_dim), as in
    # FlashMultiHeadAttention; SDPA takes (batch, heads, sequence, head_dim).
    Q, K, V = (
        torch.randn(
            shape.num_heads,
            shape.context_length,
            head_dim,
            device=DEVICE,
            dtype=DTYPE,
            requires_grad=True,
        )
        for _ in range(3)
    )
    grad_output = torch.randn_like(Q)
    sdpa_Q, sdpa_K, sdpa_V = (M.detach().unsqueeze(0).requires_grad_() for M in (Q, K, V))

    def sdpa_forward() -> torch.Tensor:
        # The backend chosen in forward also selects the backward kernel.
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            return F.scaled_dot_product_attention(sdpa_Q, sdpa_K, sdpa_V, is_causal=True)

    forward_flops = 2 * causal_attention_matmul_flops(shape)
    return measure_forward_and_backward(
        ATTENTION,
        "triton flash attention (project)",
        lambda: FlashAttentionTriton.apply(Q, K, V, True),
        (Q, K, V),
        grad_output,
        timer,
        forward_flops,
    ) + measure_forward_and_backward(
        ATTENTION,
        "pytorch sdpa, flash backend",
        sdpa_forward,
        (sdpa_Q, sdpa_K, sdpa_V),
        grad_output.unsqueeze(0),
        timer,
        forward_flops,
        reference=True,
    )


def benchmark_linear_layers(shape: ModelShape, timer: Timer) -> list[Measurement]:
    """The block's matrix multiplications through the model's own Linear module."""
    layers = {
        QKVO_PROJECTION: (shape.d_model, shape.d_model),
        SWIGLU_UP: (shape.d_model, shape.d_ff),
        SWIGLU_DOWN: (shape.d_ff, shape.d_model),
    }
    measurements = []
    for component, (d_in, d_out) in layers.items():
        # FSDP gathers BF16 compute weights; initialize in FP32 and cast to match.
        linear = Linear(d_in, d_out, device=DEVICE).to(DTYPE)
        x = torch.randn(
            1, shape.context_length, d_in, device=DEVICE, dtype=DTYPE, requires_grad=True
        )
        grad_output = torch.randn(1, shape.context_length, d_out, device=DEVICE, dtype=DTYPE)
        measurements += measure_forward_and_backward(
            component,
            "cs336_basics Linear (einsum)",
            partial(linear, x),
            (x, linear.weight),
            grad_output,
            timer,
            forward_flops=2 * shape.context_length * d_in * d_out,
        )
    return measurements


def benchmark_memory_bound_layers(shape: ModelShape, timer: Timer) -> list[Measurement]:
    """RMSNorm, RoPE, and the SwiGLU gate, with the dtypes of the FSDP run.

    FSDP casts only Linear and Embedding weights to BF16, so RMSNorm weights and
    the RoPE tables stay in FP32 while activations are BF16.
    """
    hidden = torch.randn(
        1, shape.context_length, shape.d_model, device=DEVICE, dtype=DTYPE, requires_grad=True
    )
    grad_hidden = torch.randn_like(hidden)

    norm = RMSNorm(shape.d_model, device=DEVICE)
    measurements = measure_forward_and_backward(
        RMSNORM,
        "cs336_basics RMSNorm",
        partial(norm, hidden),
        (hidden, norm.weight),
        grad_hidden,
        timer,
        forward_bytes=tensor_bytes(hidden, norm(hidden).detach()),
    )

    rope = RotaryPositionalEmbedding(
        10_000.0, shape.d_model // shape.num_heads, shape.context_length, DEVICE
    )

    def rope_forward() -> torch.Tensor:
        # As in FlashMultiHeadAttention: a strided head view of a projection
        # output, with token_positions left as None.
        heads = rearrange(hidden, "B N (H Dh) -> B H N Dh", H=shape.num_heads)
        return rope(heads)

    rope_output = rope_forward().detach()
    measurements += measure_forward_and_backward(
        ROPE,
        "cs336_basics RotaryPositionalEmbedding",
        rope_forward,
        (hidden,),
        torch.randn_like(rope_output),
        timer,
        forward_bytes=tensor_bytes(hidden, rope_output),
    )

    gate, up = (
        torch.randn(
            1, shape.context_length, shape.d_ff, device=DEVICE, dtype=DTYPE, requires_grad=True
        )
        for _ in range(2)
    )
    gated = (silu(gate) * up).detach()
    measurements += measure_forward_and_backward(
        SWIGLU_GATE,
        "cs336_basics silu(w1 x) * w3 x",
        lambda: silu(gate) * up,
        (gate, up),
        torch.randn_like(gated),
        timer,
        forward_bytes=tensor_bytes(gate, up, gated),
    )
    return measurements


def benchmark_lm_head_loss(shape: ModelShape, timer: Timer) -> list[Measurement]:
    """The fused LM head and summed cross-entropy, against materialized logits."""
    hidden = torch.randn(
        1, shape.context_length, shape.d_model, device=DEVICE, dtype=DTYPE, requires_grad=True
    )
    weight = (
        torch.randn(shape.vocab_size, shape.d_model, device=DEVICE, dtype=DTYPE) * 0.02
    ).requires_grad_()
    targets = torch.randint(shape.vocab_size, (1, shape.context_length), device=DEVICE)

    def materialized_forward() -> torch.Tensor:
        logits = einsum(hidden, weight, "... d_in, d_out d_in -> ... d_out")
        return F.cross_entropy(
            logits.reshape(-1, shape.vocab_size), targets.reshape(-1), reduction="sum"
        )

    forward_flops = lm_head_forward_flops(shape)
    return measure_forward_and_backward(
        LM_HEAD_LOSS,
        "fused triton (project)",
        lambda: FusedLMHeadCrossEntropyFunction.apply(hidden, weight, targets),
        (hidden, weight),
        None,
        timer,
        forward_flops,
    ) + measure_forward_and_backward(
        LM_HEAD_LOSS,
        "einsum logits + F.cross_entropy",
        materialized_forward,
        (hidden, weight),
        None,
        timer,
        forward_flops,
        reference=True,
    )


def rank_parameter_shard(shape: ModelShape, world_size: int) -> list[torch.nn.Parameter]:
    """FP32 parameters with random gradients, shaped like one FSDP rank's state.

    FSDP pads each 2-D weight to a multiple of world_size and keeps one flat
    shard per rank; RMSNorm weights stay whole on every rank.
    """
    params = []
    for param_shape in parameter_shapes(shape):
        numel = math.prod(param_shape)
        local_numel = math.ceil(numel / world_size) if len(param_shape) == 2 else numel
        param = torch.nn.Parameter(torch.randn(local_numel, device=DEVICE))
        param.grad = torch.randn_like(param)
        params.append(param)
    return params


def benchmark_optimizers(shape: ModelShape, world_size: int, timer: Timer) -> list[Measurement]:
    """One optimizer step over the parameter shard owned by one FSDP rank."""

    def measure(implementation: str, optimizer_cls: type[torch.optim.Optimizer]) -> Measurement:
        # Parameters, gradients, and moments are freed when this call returns.
        params = rank_parameter_shard(shape, world_size)
        optimizer = optimizer_cls(params)
        # The timer's warmup call also creates the lazily initialized moments.
        ms = timer(optimizer.step)
        local_params = sum(p.numel() for p in params)
        # Each element reads p, grad, m, v and writes p, m, v in FP32.
        return Measurement(OPTIMIZER, implementation, STEP, ms, min_bytes=7 * 4 * local_params)

    measurements = [measure(FUSED_ADAMW, FusedAdamW)]
    torch.cuda.empty_cache()
    measurements.append(measure(ADAMW, AdamW))
    torch.cuda.empty_cache()
    return measurements


def print_measurements(measurements: list[Measurement], gpu: GPUSpec) -> None:
    header = (
        f"{'component':<26}{'implementation':<40}{'mode':<18}"
        f"{'time_ms':>11}{'TFLOP/s':>10}{'%peak':>8}{'GB/s':>9}{'%bw':>7}"
    )
    print(header)
    print("-" * len(header))
    for m in measurements:
        row = f"{m.component:<26}{m.implementation:<40}{m.mode:<18}"
        if m.ms is None:
            print(row + f"{'OOM':>11}")
            continue
        row += f"{m.ms:>11.3f}"
        if m.flops is not None:
            tflops = m.flops / (m.ms / 1_000) / 1e12
            row += f"{tflops:>10.1f}{100 * tflops * 1e12 / gpu.peak_bf16_flops:>7.1f}%"
        else:
            row += " " * 18
        if m.min_bytes is not None:
            gbps = m.min_bytes / (m.ms / 1_000) / 1e9
            row += f"{gbps:>9.0f}{100 * gbps * 1e9 / gpu.memory_bandwidth:>6.1f}%"
        print(row)


def write_csv(measurements: list[Measurement], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["component", "implementation", "mode", "time_ms", "model_flops", "min_bytes"]
        )
        for m in measurements:
            writer.writerow(
                [m.component, m.implementation, m.mode, m.ms, m.flops, m.min_bytes]
            )


def print_step_budget(
    measurements: list[Measurement],
    shape: ModelShape,
    recomputed_layers: int,
    optimizer: str,
    step_ms: float | None,
) -> None:
    """Add up the project's component times into one rank's step compute budget.

    Each layer runs forward + backward once, and each checkpointed layer runs
    one extra forward during backward. External baselines are left out because
    the 8B runners do not execute them.
    """
    times = {
        (m.component, m.mode): m.ms
        for m in measurements
        if not m.reference and m.component != OPTIMIZER
    }
    optimizer_implementation = FUSED_ADAMW if optimizer == "fused-adamw" else ADAMW
    optimizer_ms = next(
        m.ms for m in measurements if m.implementation == optimizer_implementation
    )

    def block_ms(mode: str) -> float | None:
        parts = [times.get((component, mode)) for component in BLOCK_COMPONENT_COUNTS]
        if any(part is None for part in parts):
            return None
        return sum(count * part for count, part in zip(BLOCK_COMPONENT_COUNTS.values(), parts))

    block_forward_backward = block_ms(FORWARD_BACKWARD)
    block_forward = block_ms(FORWARD)
    final_norm = times.get((RMSNORM, FORWARD_BACKWARD))
    lm_head_loss = times.get((LM_HEAD_LOSS, FORWARD_BACKWARD))
    if None in (block_forward_backward, block_forward, final_norm, lm_head_loss, optimizer_ms):
        print("step budget unavailable: a required component did not fit in memory")
        return

    rows = [
        (
            f"{shape.num_layers} blocks, forward + backward",
            shape.num_layers * block_forward_backward,
        ),
        (f"{recomputed_layers} recomputed block forwards", recomputed_layers * block_forward),
        ("final rmsnorm, forward + backward", final_norm),
        ("lm head + loss, forward + backward", lm_head_loss),
        (f"optimizer step ({optimizer})", optimizer_ms),
    ]
    budget_ms = sum(ms for _, ms in rows)
    for label, ms in rows:
        print(f"{label:<44}{ms:>12.1f} ms{100 * ms / budget_ms:>8.1f}%")
    print(f"{'component compute budget':<44}{budget_ms:>12.1f} ms")
    if step_ms is not None:
        unaccounted = step_ms - budget_ms
        print(f"{'measured step time':<44}{step_ms:>12.1f} ms")
        print(
            f"{'outside the timed components':<44}{unaccounted:>12.1f} ms"
            f"{100 * unaccounted / step_ms:>8.1f}% of step"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", choices=sorted(GPU_SPECS), default="a100")
    parser.add_argument(
        "--world-size",
        type=int,
        default=4,
        help="FSDP world size that determines the optimizer's parameter shard",
    )
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
        help="checkpointed layers in the step budget "
        "(default: TransformerLM's rule, every layer except each sixteenth)",
    )
    parser.add_argument(
        "--optimizer", choices=["adamw", "fused-adamw"], default="fused-adamw"
    )
    parser.add_argument(
        "--step-ms",
        type=float,
        default=None,
        help="measured distributed step time to compare with the budget",
    )
    parser.add_argument("--warmup-ms", type=int, default=100)
    parser.add_argument("--rep-ms", type=int, default=1_000)
    parser.add_argument("--output-csv", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.world_size <= 0:
        raise ValueError("world_size must be positive")
    if args.d_model % args.num_heads != 0:
        raise ValueError("d_model must be divisible by num_heads")
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
    timer = make_timer(args.warmup_ms, args.rep_ms)

    print(f"device={torch.cuda.get_device_name(DEVICE)}")
    print(f"reference_gpu={gpu.name}")
    print(f"peak_bf16_tflops={gpu.peak_bf16_flops / 1e12:.1f}")
    print(f"memory_bandwidth_gb_per_s={gpu.memory_bandwidth / 1e9:.0f}")
    print(f"context_length={shape.context_length}")
    print(f"d_model={shape.d_model}")
    print(f"d_ff={shape.d_ff}")
    print(f"num_heads={shape.num_heads}")
    print(f"vocab_size={shape.vocab_size}")
    print(f"optimizer_world_size={args.world_size}")
    print()

    torch.manual_seed(0)
    measurements = []
    for benchmark in (
        benchmark_attention,
        benchmark_linear_layers,
        benchmark_memory_bound_layers,
        benchmark_lm_head_loss,
    ):
        measurements += benchmark(shape, timer)
        torch.cuda.empty_cache()
    measurements += benchmark_optimizers(shape, args.world_size, timer)

    print_measurements(measurements, gpu)
    print()
    print_step_budget(measurements, shape, recomputed_layers, args.optimizer, args.step_ms)

    if args.output_csv is not None:
        write_csv(measurements, args.output_csv)
        print(f"\ncsv={args.output_csv}")


if __name__ == "__main__":
    main()
