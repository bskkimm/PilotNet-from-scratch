"""Reproducibility configuration and run metadata."""

import os
import random
from dataclasses import asdict
from typing import Any

import numpy
import torch

from pilotnet.data.preprocessing import PreprocessConfig


def seed_everything(seed: int, deterministic: bool) -> dict[str, object]:
    """Seed supported random generators and configure Torch determinism."""
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    return {"seed": seed, "deterministic": deterministic}


def build_run_manifest(
    *,
    preprocessing: PreprocessConfig,
    sampling: dict[str, object],
    reproducibility: dict[str, object],
    **metadata: Any,
) -> dict[str, object]:
    """Build JSON-serializable metadata for a training run."""
    return {
        "preprocessing": asdict(preprocessing),
        "sampling": sampling,
        "reproducibility": reproducibility,
        **metadata,
    }
