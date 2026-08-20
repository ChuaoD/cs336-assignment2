from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path

import torch
from torch.utils.checkpoint import checkpoint

from cs336_basics.model import TransformerLM
from cs336_basics.utils import cross_entropy_loss


MIB = 1024**2


def checkpointed_forward(
    model: TransformerLM,
    token_ids: torch.Tensor,
    token_positions: torch.Tensor,
    checkpoint_block_size: int,
) -> torch.Tensor:
    """Run the Transformer with one non-nested checkpoint level."""
    if checkpoint_block_size < 1:
        raise ValueError("checkpoint_block_size must be at least 1")

    hidden = model.token_embeddings(token_ids)
    num_layers = len(model.layers)

    for start in range(0, num_layers, checkpoint_block_size):
        end = min(start + checkpoint_block_size, num_layers)

        def run_segment(
            segment_input: torch.Tensor,
            start_index: int = start,
            end_index: int = end,
        ) -> torch.Tensor:
            segment_output = segment_input
            for layer_index in range(start_index, end_index):
                segment_output = model.layers[layer_index](
                    segment_output,
                    token_positions,
                )
            return segment_output

        hidden = checkpoint(run_segment, hidden, use_reentrant=False)

    hidden = model.ln_final(hidden)
    return model.lm_head(hidden)


def profile_peak_memory(
    *,
    checkpoint_block_size: int,
    output_path: Path,
    size: str = "xl",
    d_model: int = 2560,
    d_ff: int = 10240,
    num_layers: int = 32,
    num_heads: int = 32,
    batch_size: int = 4,
    context_length: int = 2048,
    vocab_size: int = 10_000,
    theta: float = 10_000.0,
    warmup_steps: int = 1,
    seed: int = 0,
    device: str = "cuda",
) -> dict[str, object]:
    if checkpoint_block_size < 1 or checkpoint_block_size > num_layers:
        raise ValueError(
            "checkpoint_block_size must be between 1 and num_layers "
            f"(received {checkpoint_block_size}, num_layers={num_layers})"
        )
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")

    cuda_device = torch.device(device)
    if cuda_device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This profiler requires an available CUDA device.")

    # Slurm exposes the allocated GPU through CUDA_VISIBLE_DEVICES. A bare
    # torch.device("cuda") has no explicit index, but torch.cuda.set_device()
    # requires one in recent PyTorch versions, so resolve it to the current
    # process-local CUDA index (normally cuda:0 under Slurm).
    if cuda_device.index is None:
        cuda_device = torch.device("cuda", torch.cuda.current_device())

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.set_device(cuda_device.index)
    torch.cuda.reset_peak_memory_stats(cuda_device)

    num_segments = math.ceil(num_layers / checkpoint_block_size)
    metrics: dict[str, object] = {
        "status": "running",
        "error": None,
        "size": size,
        "checkpoint_block_size": checkpoint_block_size,
        "num_checkpoint_segments": num_segments,
        "batch_size": batch_size,
        "context_length": context_length,
        "d_model": d_model,
        "d_ff": d_ff,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "vocab_size": vocab_size,
        "warmup_steps": warmup_steps,
        "seed": seed,
    }

    baseline_allocated: int | None = None
    peak_allocated: int | None = None
    peak_reserved: int | None = None
    end_allocated: int | None = None

    try:
        model = TransformerLM(
            vocab_size,
            context_length,
            num_layers,
            d_model,
            num_heads,
            d_ff,
            theta,
            cuda_device,
        )
        model.train()

        input_batch = torch.randint(
            0,
            vocab_size,
            (batch_size, context_length),
            dtype=torch.long,
            device=cuda_device,
        )
        target_batch = torch.randint(
            0,
            vocab_size,
            (batch_size, context_length),
            dtype=torch.long,
            device=cuda_device,
        )
        token_positions = torch.arange(context_length, device=cuda_device)

        boundary_tensor_bytes = (
            batch_size
            * context_length
            * d_model
            * model.token_embeddings.weight.element_size()
        )
        metrics.update(
            {
                "gpu_name": torch.cuda.get_device_name(cuda_device),
                "parameter_count": sum(
                    parameter.numel() for parameter in model.parameters()
                ),
                "parameter_dtype": str(model.token_embeddings.weight.dtype),
                "boundary_tensor_bytes": boundary_tensor_bytes,
                "boundary_tensor_mib": boundary_tensor_bytes / MIB,
                "estimated_all_boundaries_mib": (
                    num_segments * boundary_tensor_bytes / MIB
                ),
            }
        )

        # Warmup is outside the measured region. It initializes CUDA kernels and
        # verifies that forward plus checkpoint recomputation can complete.
        for _ in range(warmup_steps):
            model.zero_grad(set_to_none=True)
            logits = checkpointed_forward(
                model,
                input_batch,
                token_positions,
                checkpoint_block_size,
            )
            loss = cross_entropy_loss(logits, target_batch)
            loss.backward()
            torch.cuda.synchronize(cuda_device)
            del logits, loss

        model.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(cuda_device)

        # The baseline contains persistent model/input storage. Subtracting it
        # from the peak is useful for comparison, but also includes gradients
        # created during backward, so it is called a step increment rather than
        # pure activation memory.
        baseline_allocated = torch.cuda.memory_allocated(cuda_device)
        torch.cuda.reset_peak_memory_stats(cuda_device)

        logits = checkpointed_forward(
            model,
            input_batch,
            token_positions,
            checkpoint_block_size,
        )
        loss = cross_entropy_loss(logits, target_batch)
        loss.backward()
        torch.cuda.synchronize(cuda_device)

        peak_allocated = torch.cuda.max_memory_allocated(cuda_device)
        peak_reserved = torch.cuda.max_memory_reserved(cuda_device)
        end_allocated = torch.cuda.memory_allocated(cuda_device)
        metrics["status"] = "ok"
    except torch.OutOfMemoryError as error:
        metrics["status"] = "oom"
        metrics["error"] = str(error)
        peak_allocated = torch.cuda.max_memory_allocated(cuda_device)
        peak_reserved = torch.cuda.max_memory_reserved(cuda_device)
        end_allocated = torch.cuda.memory_allocated(cuda_device)

    metrics.update(
        {
            "baseline_allocated_bytes": baseline_allocated,
            "baseline_allocated_mib": (
                baseline_allocated / MIB if baseline_allocated is not None else None
            ),
            "peak_allocated_bytes": peak_allocated,
            "peak_allocated_mib": (
                peak_allocated / MIB if peak_allocated is not None else None
            ),
            "peak_reserved_bytes": peak_reserved,
            "peak_reserved_mib": (
                peak_reserved / MIB if peak_reserved is not None else None
            ),
            "end_allocated_bytes": end_allocated,
            "end_allocated_mib": (
                end_allocated / MIB if end_allocated is not None else None
            ),
            "peak_step_increment_bytes": (
                peak_allocated - baseline_allocated
                if peak_allocated is not None and baseline_allocated is not None
                else None
            ),
            "peak_step_increment_mib": (
                (peak_allocated - baseline_allocated) / MIB
                if peak_allocated is not None and baseline_allocated is not None
                else None
            ),
        }
    )

    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))

    if metrics["status"] == "oom":
        raise SystemExit(1)
    return metrics


def summarize_results(result_paths: list[Path]) -> None:
    rows: list[dict[str, object]] = []
    for result_path in result_paths:
        if not result_path.is_file():
            continue
        rows.append(json.loads(result_path.read_text(encoding="utf-8")))

    if not rows:
        print("No result JSON files were produced.")
        return

    rows.sort(key=lambda row: int(row["checkpoint_block_size"]))
    headers = (
        "block_size",
        "segments",
        "status",
        "peak_allocated_mib",
        "step_increment_mib",
        "peak_reserved_mib",
    )
    print("\t".join(headers))
    for row in rows:
        values = (
            str(row["checkpoint_block_size"]),
            str(row["num_checkpoint_segments"]),
            str(row["status"]),
            _format_mib(row.get("peak_allocated_mib")),
            _format_mib(row.get("peak_step_increment_mib")),
            _format_mib(row.get("peak_reserved_mib")),
        )
        print("\t".join(values))


def _format_mib(value: object) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.2f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure peak CUDA memory for one non-nested gradient-checkpoint "
            "block size. Run neighboring sizes in separate processes."
        )
    )
    parser.add_argument("--checkpoint_block_size", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--summarize",
        type=Path,
        nargs="+",
        help="Print a tab-separated comparison of existing result JSON files.",
    )
    parser.add_argument("--size", default="xl")
    parser.add_argument("--d_model", type=int, default=2560)
    parser.add_argument("--d_ff", type=int, default=10240)
    parser.add_argument("--num_layers", type=int, default=32)
    parser.add_argument("--num_heads", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--context_length", type=int, default=2048)
    parser.add_argument("--vocab_size", type=int, default=10_000)
    parser.add_argument("--theta", type=float, default=10_000.0)
    parser.add_argument("--warmup_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.summarize is None and (
        args.checkpoint_block_size is None or args.output is None
    ):
        parser.error(
            "profiling requires --checkpoint_block_size and --output; "
            "otherwise use --summarize RESULT.json [...]"
        )
    return args


def main() -> None:
    args = parse_args()
    if args.summarize is not None:
        summarize_results(args.summarize)
        return

    assert args.checkpoint_block_size is not None
    assert args.output is not None
    profile_peak_memory(
        checkpoint_block_size=args.checkpoint_block_size,
        output_path=args.output,
        size=args.size,
        d_model=args.d_model,
        d_ff=args.d_ff,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        batch_size=args.batch_size,
        context_length=args.context_length,
        vocab_size=args.vocab_size,
        theta=args.theta,
        warmup_steps=args.warmup_steps,
        seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    main()
