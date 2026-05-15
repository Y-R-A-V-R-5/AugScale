"""
src/utils/seed.py
==================
Global reproducibility utilities for Phase-2 YOLOv8 experiments.

This module ensures deterministic behavior across:

- Python runtime
- NumPy operations
- PyTorch CPU/GPU execution (if installed)

Reproducibility is critical for:
- fair model comparison (M0-M5)
- stable evaluation under augmentation
- consistent robustness ranking
"""

from __future__ import annotations

import os
import random
import numpy as np


# ---------------------------------------------------------------------
# Global Seed Setter
# ---------------------------------------------------------------------

def set_global_seed(seed: int = 42) -> None:
    """
    Set global random seed across all supported libraries.

    Parameters
    ----------
    seed : int
        Seed value used for all RNGs

    Ensures:
    - deterministic Python behavior
    - deterministic NumPy operations
    - deterministic PyTorch (if installed)
    - CUDA determinism (if GPU available)
    """

    # -------------------------------------------------------------
    # Python-level reproducibility
    # -------------------------------------------------------------

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    # -------------------------------------------------------------
    # PyTorch (optional dependency)
    # -------------------------------------------------------------

    try:
        import torch

        torch.manual_seed(seed)

        # GPU determinism
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

            # Ensure deterministic CUDA behavior
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    except ImportError:
        # PyTorch not installed  -  skip silently
        pass