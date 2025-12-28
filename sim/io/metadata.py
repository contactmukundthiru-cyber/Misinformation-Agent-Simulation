from __future__ import annotations

import platform
import subprocess
import sys
from importlib import metadata
from typing import Dict

import torch

from sim.config import SimulationConfig


def _pkg_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "unknown"


def build_run_metadata(cfg: SimulationConfig, device: torch.device) -> Dict[str, str]:
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        git_hash = "unknown"
    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": str(torch.cuda.is_available()),
        "device": str(device),
        "numpy_version": _pkg_version("numpy"),
        "pandas_version": _pkg_version("pandas"),
        "networkx_version": _pkg_version("networkx"),
        "pyarrow_version": _pkg_version("pyarrow"),
        "streamlit_version": _pkg_version("streamlit"),
        "sim_version": _pkg_version("town-misinfo-sim"),
        "git_commit": git_hash,
        "seed": str(cfg.sim.seed),
        "deterministic": str(cfg.sim.deterministic),
    }
