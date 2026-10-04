"""Capture reproducibility metadata from the actual b300-2 host."""

import json
import platform
import subprocess
import sys
from pathlib import Path

try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:
    from importlib_metadata import PackageNotFoundError, version


PACKAGES = (
    "isaacsim", "isaaclab", "isaaclab-tasks", "torch", "numpy", "h5py", "tensorboard", "matplotlib"
)


def command(args):
    try:
        result = subprocess.run(args, text=True, capture_output=True, check=True, timeout=20)
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        return f"UNAVAILABLE: {type(exc).__name__}: {exc}"


def main():
    root = Path(__file__).resolve().parents[1]
    packages = {}
    for package in PACKAGES:
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    report = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "packages": packages,
        "gpus": command(["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version", "--format=csv,noheader"]),
        "glibc": command(["ldd", "--version"]).splitlines()[0],
        "isaaclab_commit": command(["git", "-C", str(root / "third_party/IsaacLab"), "rev-parse", "HEAD"]),
    }
    try:
        import torch

        report["torch_cuda_runtime"] = torch.version.cuda
        report["torch_cuda_available"] = torch.cuda.is_available()
        report["torch_device_count"] = torch.cuda.device_count()
    except ImportError:
        report["torch_cuda_runtime"] = None
    path = root / "results/environment.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
