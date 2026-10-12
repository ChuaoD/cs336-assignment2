from __future__ import annotations

import argparse
import bisect
import re
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import mean

from cs336_systems.mfu import (
    GPU_SPECS,
    ModelShape,
    implementation_flops,
    model_flops,
)

STEP = "step"
PHASES = ("forward", "backward", "finish_gradient_synchronization", "optimizer")
RECOMPUTE = "activation checkpoint recomputation"
REGION_CATEGORIES = {
    "flash attention forward": "attention forward",
    "flash attention backward": "attention backward",
    "fused lm head loss forward": "lm head + loss forward",
    "fused lm head loss backward": "lm head + loss backward",
}
TRACKED_RANGES = (STEP, *PHASES, RECOMPUTE, *REGION_CATEGORIES)

COMMUNICATION = "nccl communication"
BLOCK_GEMMS = "block gemms"
OPTIMIZER = "optimizer"
OTHER = "other kernels"
MEMCPY = "memcpy"
MEMSET = "memset"
CATEGORY_ORDER = (
    "attention forward",
    "attention backward",
    BLOCK_GEMMS,
    "lm head + loss forward",
    "lm head + loss backward",
    OPTIMIZER,
    OTHER,
    MEMCPY,
    MEMSET,
    COMMUNICATION,
)
# Matches cuBLAS and CUTLASS kernel names, including split-K reductions.
GEMM_PATTERN = re.compile(r"gemm|xmma|cutlass|splitk", re.IGNORECASE)


@dataclass(frozen=True)
class GPUActivity:
    pid: int
    device: int
    start: int
    end: int
    name: str
    launch: int | None  # CPU timestamp of the launching runtime call
    category: str = ""


class Ranges:
    """Non-overlapping NVTX ranges with one name in one process."""

    def __init__(self, intervals: list[tuple[int, int]]) -> None:
        intervals = sorted(intervals)
        self.starts = [start for start, _ in intervals]
        self.ends = [end for _, end in intervals]

    def __len__(self) -> int:
        return len(self.starts)

    def find(self, timestamp: int | None) -> int | None:
        """Index of the range containing timestamp, or None."""
        if timestamp is None:
            return None
        index = bisect.bisect_right(self.starts, timestamp) - 1
        if index >= 0 and timestamp <= self.ends[index]:
            return index
        return None

    def count_within(self, start: int, end: int) -> int:
        """Number of ranges that begin inside [start, end]."""
        return bisect.bisect_right(self.starts, end) - bisect.bisect_left(self.starts, start)


def process_id(global_id: int) -> int:
    """Extract the process ID from an Nsight Systems global thread or process ID."""
    return (global_id >> 24) & 0xFFFFFF


def union_length(intervals: list[tuple[int, int]]) -> int:
    total = 0
    current_start = current_end = None
    for start, end in sorted(intervals):
        if current_end is None or start > current_end:
            if current_end is not None:
                total += current_end - current_start
            current_start, current_end = start, end
        else:
            current_end = max(current_end, end)
    if current_end is not None:
        total += current_end - current_start
    return total


def table_names(connection: sqlite3.Connection) -> set[str]:
    return {
        row[0]
        for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }


def load_ranges(connection: sqlite3.Connection) -> dict[int, dict[str, Ranges]]:
    """Return the tracked NVTX ranges of every process, grouped by name."""
    rows = connection.execute(
        """
        SELECT n.globalTid, n.start, n.end, COALESCE(n.text, s.value, '')
        FROM NVTX_EVENTS n
        LEFT JOIN StringIds s ON s.id = n.textId
        WHERE n.end IS NOT NULL  -- ranges only; marks have no end
        """
    )
    intervals: dict[int, dict[str, list[tuple[int, int]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for global_tid, start, end, text in rows:
        if text in TRACKED_RANGES:
            intervals[process_id(global_tid)][text].append((start, end))
    return {
        pid: {name: Ranges(intervals[pid][name]) for name in TRACKED_RANGES}
        for pid in intervals
    }


def load_kernels(connection: sqlite3.Connection) -> list[GPUActivity]:
    launches = {
        (process_id(global_tid), correlation_id): start
        for global_tid, correlation_id, start in connection.execute(
            "SELECT globalTid, correlationId, start FROM CUPTI_ACTIVITY_KIND_RUNTIME"
        )
    }
    kernels = []
    for global_pid, device, start, end, correlation_id, name in connection.execute(
        """
        SELECT k.globalPid, k.deviceId, k.start, k.end, k.correlationId, s.value
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON s.id = k.shortName
        """
    ):
        pid = process_id(global_pid)
        kernels.append(
            GPUActivity(pid, device, start, end, name, launches.get((pid, correlation_id)))
        )
    return kernels


def load_copies(connection: sqlite3.Connection, tables: set[str]) -> list[GPUActivity]:
    """Memcpy and memset activity; a memcpy's name is its copy direction."""
    copy_kinds = {}
    if "ENUM_CUDA_MEMCPY_OPER" in tables:
        copy_kinds = dict(connection.execute("SELECT id, label FROM ENUM_CUDA_MEMCPY_OPER"))
    copies = []
    if "CUPTI_ACTIVITY_KIND_MEMCPY" in tables:
        for global_pid, device, start, end, kind in connection.execute(
            """
            SELECT globalPid, deviceId, start, end, copyKind
            FROM CUPTI_ACTIVITY_KIND_MEMCPY
            """
        ):
            name = f"memcpy {copy_kinds.get(kind, kind)}"
            copies.append(
                GPUActivity(process_id(global_pid), device, start, end, name, None, MEMCPY)
            )
    if "CUPTI_ACTIVITY_KIND_MEMSET" in tables:
        for global_pid, device, start, end in connection.execute(
            "SELECT globalPid, deviceId, start, end FROM CUPTI_ACTIVITY_KIND_MEMSET"
        ):
            copies.append(
                GPUActivity(process_id(global_pid), device, start, end, "memset", None, MEMSET)
            )
    return copies


def categorize(kernel: GPUActivity, ranges: dict[str, Ranges]) -> str:
    if kernel.name.startswith("nccl"):
        return COMMUNICATION
    for region, category in REGION_CATEGORIES.items():
        if ranges[region].find(kernel.launch) is not None:
            return category
    if ranges["optimizer"].find(kernel.launch) is not None:
        return OPTIMIZER
    if GEMM_PATTERN.search(kernel.name):
        return BLOCK_GEMMS
    return OTHER


def print_table(header: tuple[str, ...], rows: list[tuple], widths: tuple[int, ...]) -> None:
    """Print rows with the first column left-aligned and the rest right-aligned."""

    def format_row(values: tuple) -> str:
        cells = [f"{values[0]:<{widths[0]}}"]
        cells += [f"{value:>{width}}" for value, width in zip(values[1:], widths[1:])]
        return "".join(cells)

    print(format_row(header))
    print("-" * sum(widths))
    for row in rows:
        print(format_row(row))


def analyze(database: Path, shape: ModelShape, local_batch_size: int, gpu_name: str) -> None:
    connection = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)
    tables = table_names(connection)
    ranges = load_ranges(connection)
    activities = load_kernels(connection)
    activities = [
        replace(kernel, category=categorize(kernel, ranges[kernel.pid]))
        for kernel in activities
        if kernel.pid in ranges
    ]
    activities += [copy for copy in load_copies(connection, tables) if copy.pid in ranges]

    # Group GPU activity by (process, step). Each step synchronizes before its
    # range closes, so a step's GPU work lies inside its CPU range.
    pids = sorted(pid for pid in ranges if len(ranges[pid][STEP]) > 0)
    if not pids:
        raise RuntimeError("No NVTX 'step' ranges found; was profile_8B.py profiled?")
    by_step: dict[tuple[int, int], list[GPUActivity]] = defaultdict(list)
    for activity in activities:
        step = ranges[activity.pid][STEP].find(activity.start)
        if step is not None:
            by_step[(activity.pid, step)].append(activity)
    step_keys = [(pid, step) for pid in pids for step in range(len(ranges[pid][STEP]))]
    device_of = {
        pid: Counter(a.device for a in activities if a.pid == pid).most_common(1)[0][0]
        for pid in pids
    }

    def step_window(key: tuple[int, int]) -> tuple[int, int]:
        pid, step = key
        return ranges[pid][STEP].starts[step], ranges[pid][STEP].ends[step]

    step_ms = {key: (step_window(key)[1] - step_window(key)[0]) / 1e6 for key in step_keys}
    mean_step_ms = mean(step_ms.values())

    # Layer counts come from the trace: one FlashAttention backward per layer,
    # one recomputation range per checkpointed layer.
    layer_counts = {
        (
            ranges[pid]["flash attention backward"].count_within(*step_window((pid, step))),
            ranges[pid][RECOMPUTE].count_within(*step_window((pid, step))),
        )
        for pid, step in step_keys
    }
    if len(layer_counts) != 1:
        raise RuntimeError(f"Steps disagree on (layers, recomputed layers): {layer_counts}")
    num_layers, recomputed_layers = layer_counts.pop()
    shape = replace(shape, num_layers=num_layers)

    print(f"database={database.name}")
    print(f"ranks={len(pids)}")
    print(f"profiled_steps={len(step_keys) // len(pids)}")
    print(f"layers_from_trace={num_layers}")
    print(f"recomputed_layers_from_trace={recomputed_layers}")
    print(f"local_batch_size={local_batch_size}")
    print(f"context_length={shape.context_length}")
    print()

    # 1. Step and phase time.
    print("== Step time per rank (CPU range of each profiled step) ==")
    print_table(
        ("rank (GPU)", "mean step ms"),
        [
            (
                f"GPU {device_of[pid]}",
                f"{mean(step_ms[(pid, step)] for step in range(len(ranges[pid][STEP]))):.1f}",
            )
            for pid in pids
        ],
        (14, 16),
    )
    print()

    def gpu_span_ms(names: tuple[str, ...]) -> dict[str, float]:
        """Mean per-step GPU span of kernels launched inside ranges with these names."""
        spans: dict[tuple, list[int]] = {}
        for key in step_keys:
            for kernel in by_step[key]:
                for name in names:
                    instance = ranges[key[0]][name].find(kernel.launch)
                    if instance is None:
                        continue
                    span_key = (key, name, instance)
                    if span_key not in spans:
                        spans[span_key] = [kernel.start, kernel.end]
                    else:
                        spans[span_key][0] = min(spans[span_key][0], kernel.start)
                        spans[span_key][1] = max(spans[span_key][1], kernel.end)
        totals: dict[str, float] = defaultdict(float)
        for (_, name, _), (start, end) in spans.items():
            totals[name] += (end - start) / 1e6
        return {name: totals[name] / len(step_keys) for name in names}

    phase_ms = gpu_span_ms((*PHASES, RECOMPUTE))
    print("== Phase time (GPU span of kernels launched in each NVTX range) ==")
    phase_rows = [
        (phase, f"{phase_ms[phase]:.1f}", f"{100 * phase_ms[phase] / mean_step_ms:.1f}%")
        for phase in PHASES
    ]
    phase_rows.insert(
        2,
        (
            "  of which recomputation",
            f"{phase_ms[RECOMPUTE]:.1f}",
            f"{100 * phase_ms[RECOMPUTE] / mean_step_ms:.1f}%",
        ),
    )
    phase_rows.append(("step", f"{mean_step_ms:.1f}", "100.0%"))
    print_table(("phase", "ms/step", "% step"), phase_rows, (36, 12, 10))
    print()

    # 2. Kernel time by category, with executed FLOP rates.
    executed = implementation_flops(shape, recomputed_layers).scaled(local_batch_size)
    category_flops = {
        "attention forward": executed.attention_forward,
        "attention backward": executed.attention_backward,
        BLOCK_GEMMS: executed.block_matmuls,
        "lm head + loss forward": executed.lm_head_forward,
        "lm head + loss backward": executed.lm_head_backward,
    }
    category_ns: dict[str, int] = Counter()
    for key in step_keys:
        for activity in by_step[key]:
            category_ns[activity.category] += activity.end - activity.start
    peak_flops = GPU_SPECS[gpu_name].peak_bf16_flops
    category_rows = []
    for category in CATEGORY_ORDER:
        ms = category_ns[category] / 1e6 / len(step_keys)
        row = [category, f"{ms:.1f}", f"{100 * ms / mean_step_ms:.1f}%", "", ""]
        if category in category_flops and ms > 0:
            tflops = category_flops[category] / (ms / 1_000) / 1e12
            row[3] = f"{tflops:.1f}"
            row[4] = f"{100 * tflops * 1e12 / peak_flops:.1f}%"
        category_rows.append(tuple(row))
    print("== GPU time by kernel category (summed durations; streams can overlap) ==")
    print_table(
        ("category", "ms/step", "% step", "exec TFLOP/s", "% peak"),
        category_rows,
        (26, 12, 10, 14, 10),
    )
    print()

    # 3. Timeline utilization per device.
    timeline_rows = []
    totals = defaultdict(list)
    for pid in pids:
        per_step = defaultdict(list)
        for step in range(len(ranges[pid][STEP])):
            window_start, window_end = step_window((pid, step))
            communication, computation = [], []
            for activity in by_step[(pid, step)]:
                interval = (max(activity.start, window_start), min(activity.end, window_end))
                if activity.category == COMMUNICATION:
                    communication.append(interval)
                else:
                    computation.append(interval)
            window = window_end - window_start
            busy = union_length(communication + computation)
            compute_busy = union_length(computation)
            # Communication time with no other GPU work alongside it.
            exposed = busy - compute_busy
            per_step["compute busy"].append(compute_busy / window)
            per_step["communication busy"].append(union_length(communication) / window)
            per_step["exposed communication"].append(exposed / window)
            per_step["idle"].append((window - busy) / window)
        row = [f"GPU {device_of[pid]}"]
        for metric in ("compute busy", "communication busy", "exposed communication", "idle"):
            value = mean(per_step[metric])
            totals[metric].append(value)
            row.append(f"{100 * value:.1f}%")
        timeline_rows.append(tuple(row))
    timeline_rows.append(
        ("mean",)
        + tuple(
            f"{100 * mean(totals[metric]):.1f}%"
            for metric in ("compute busy", "communication busy", "exposed communication", "idle")
        )
    )
    print("== Timeline utilization (fraction of step wall time) ==")
    print_table(
        ("rank (GPU)", "compute busy", "comm busy", "exposed comm", "idle"),
        timeline_rows,
        (14, 15, 13, 15, 10),
    )
    print("Exposed communication includes NCCL kernels waiting for slower ranks.")
    print()

    # Utilization of the profiled steps, which include tracing overhead.
    model_total = model_flops(shape).total * local_batch_size
    step_seconds = mean_step_ms / 1_000
    print(f"profiled_mfu={model_total / step_seconds / peak_flops:.4f}")
    print(f"profiled_hfu={executed.total / step_seconds / peak_flops:.4f}")
    print()

    # 4. Most expensive kernels and memory copies.
    kernel_ns: dict[tuple[str, str], int] = Counter()
    kernel_calls: dict[tuple[str, str], int] = Counter()
    for key in step_keys:
        for activity in by_step[key]:
            kernel_ns[(activity.name, activity.category)] += activity.end - activity.start
            kernel_calls[(activity.name, activity.category)] += 1
    top_rows = []
    for (name, category), ns in kernel_ns.most_common(20):
        calls = kernel_calls[(name, category)] / len(step_keys)
        ms = ns / 1e6 / len(step_keys)
        top_rows.append(
            (
                name[:60],
                category,
                f"{ms:.1f}",
                f"{calls:.0f}",
                f"{1_000 * ms / calls:.1f}",
            )
        )
    print("== Top 20 kernels and copies by GPU time ==")
    print_table(
        ("name", "category", "ms/step", "calls/step", "mean us"),
        top_rows,
        (62, 26, 10, 12, 10),
    )
    print()

    # Host-to-device copies inside a step usually mean a CPU tensor reached
    # GPU code, which can also stall the host.
    copy_rows = [
        (
            name,
            f"{ns / 1e6 / len(step_keys):.2f}",
            f"{kernel_calls[(name, category)] / len(step_keys):.0f}",
        )
        for (name, category), ns in kernel_ns.most_common()
        if category in (MEMCPY, MEMSET)
    ]
    print("== Memory copies and sets ==")
    print_table(("kind", "ms/step", "calls/step"), copy_rows, (36, 12, 12))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("database", type=Path)
    parser.add_argument("--gpu", choices=sorted(GPU_SPECS), default="a100")
    parser.add_argument("--local-batch-size", type=int, default=1)
    parser.add_argument("--context-length", type=int, default=32_768)
    parser.add_argument("--vocab-size", type=int, default=151_936)
    parser.add_argument("--d-model", type=int, default=4_096)
    parser.add_argument("--d-ff", type=int, default=11_008)
    parser.add_argument("--num-heads", type=int, default=32)
    args = parser.parse_args()

    shape = ModelShape(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        d_ff=args.d_ff,
        num_heads=args.num_heads,
    )
    analyze(args.database, shape, args.local_batch_size, args.gpu)


if __name__ == "__main__":
    main()
