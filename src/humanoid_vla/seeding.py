"""Global seeding for reproducible training runs."""

from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and (if available) PyTorch RNGs.

    Args:
        seed: base seed, recorded in every checkpoint.
        deterministic: also force deterministic cuDNN kernels. Slower; use for
            debugging exact reproducibility rather than normal training.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # Non-deterministic but fast; benchmark picks the best conv kernels.
        torch.backends.cudnn.benchmark = True


def worker_init_fn(worker_id: int) -> None:
    """DataLoader worker init that derives a distinct seed per worker."""
    import torch

    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
