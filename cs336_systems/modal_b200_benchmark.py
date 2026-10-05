from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import modal


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSIGNMENT1_PACKAGE = PROJECT_ROOT.parent / "cs336-assignment1" / "cs336_basics"
REMOTE_PROJECT_ROOT = "/root/cs336"


image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "torch==2.11.0",
        "triton==3.6.0",
        "einops==0.8.2",
        "jaxtyping==0.3.9",
        "numpy==2.4.4",
    )
    .env(
        {
            "PYTHONPATH": REMOTE_PROJECT_ROOT,
            "PYTHONUNBUFFERED": "1",
            "PYTORCH_KERNEL_CACHE_PATH": "/tmp/torch-kernels",
            "TRITON_CACHE_DIR": "/tmp/triton-cache",
        }
    )
    .add_local_dir(
        str(PROJECT_ROOT / "cs336_systems"),
        remote_path=f"{REMOTE_PROJECT_ROOT}/cs336_systems",
    )
    .add_local_dir(
        str(ASSIGNMENT1_PACKAGE),
        remote_path=f"{REMOTE_PROJECT_ROOT}/cs336_basics",
    )
)

app = modal.App("cs336-section9-b200-benchmark")


def _benchmark_arguments(profile: str) -> list[str]:
    common = [
        "--world-size",
        "2",
        "--global-batch-size",
        "2",
        "--warmup-steps",
        "1",
    ]

    if profile == "tiny":
        return common + [
            "--context-length",
            "1024",
            "--num-layers",
            "2",
            "--measured-steps",
            "1",
            "--activation-checkpointing",
        ]

    if profile == "full":
        return common + [
            "--context-length",
            "32768",
            "--num-layers",
            "34",
            "--measured-steps",
            "3",
            "--activation-checkpointing",
        ]

    raise ValueError(f"Unknown benchmark profile: {profile!r}")


def _benchmark_command(profile: str) -> list[str]:
    if profile == "section9":
        return [
            sys.executable,
            "-m",
            "cs336_systems.benchmark_8B_section9",
            "--warmup-ms",
            "10000",
            "--rep-ms",
            "30000",
            "--activation-checkpointing",
        ]

    return [
        sys.executable,
        "-m",
        "cs336_systems.benchmark_8B",
        *_benchmark_arguments(profile),
    ]


@app.function(
    image=image,
    gpu="B200:2",
    cpu=8.0,
    memory=32_768,
    timeout=3_600,
)
def run_benchmark(profile: str) -> None:
    import torch
    import triton

    if profile not in {"smoke", "tiny", "full", "section9"}:
        raise ValueError("profile must be one of: smoke, tiny, full, section9")

    print(f"torch_version={torch.__version__}")
    print(f"triton_version={triton.__version__}")
    print(f"cuda_runtime={torch.version.cuda}")
    print(f"cuda_available={torch.cuda.is_available()}")
    print(f"cuda_device_count={torch.cuda.device_count()}")
    print(f"nccl_available={torch.distributed.is_nccl_available()}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in the Modal container")
    if torch.cuda.device_count() != 2:
        raise RuntimeError(
            f"Expected exactly 2 visible B200 GPUs, got {torch.cuda.device_count()}"
        )

    for device_index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(device_index)
        print(f"gpu_{device_index}_name={properties.name}")
        print(
            f"gpu_{device_index}_memory_gib="
            f"{properties.total_memory / 1024**3:.3f}"
        )
        print(
            f"gpu_{device_index}_capability="
            f"{properties.major}.{properties.minor}"
        )

    if profile == "smoke":
        print("smoke_check=passed")
        return

    command = _benchmark_command(profile)
    print("benchmark_command=" + " ".join(command))
    subprocess.run(
        command,
        cwd=REMOTE_PROJECT_ROOT,
        check=True,
    )


@app.local_entrypoint()
def main(profile: str = "smoke") -> None:
    """Run a smoke, diagnostic, or Section 9 two-B200 profile."""
    if not ASSIGNMENT1_PACKAGE.is_dir():
        raise RuntimeError(
            "Expected Assignment 1 at ../cs336-assignment1/cs336_basics "
            "relative to this repository."
        )
    if profile not in {"smoke", "tiny", "full", "section9"}:
        raise ValueError(
            "--profile must be one of: smoke, tiny, full, section9"
        )
    run_benchmark.remote(profile)
