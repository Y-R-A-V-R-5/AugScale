"""
src/augmentations/policies.py
==============================
Defines the 6 training augmentation policies (M0 - M5) for Phase-2 robustness study.

Each policy is a deterministic function:

    policy(image: np.ndarray, seed: int) -> np.ndarray

Key design constraints
----------------------
- NO hardcoded augmentation parameters
- ALL parameters come from registry.py
- Deterministic behavior via seed control
- Same augmentation space used in evaluation pipeline

Policy overview
---------------
M0  Baseline           -> no augmentation
M1  Geometric          -> perspective / affine / hflip
M2  Weather            -> fog / rain / haze
M3  Blur / Noise       -> gaussian blur / motion blur / noise
M4  Illumination       -> brightness / contrast / gamma
M5  Mixed Robust       -> uniform sampling across all families
"""

from __future__ import annotations

import random
import numpy as np
from typing import Callable, Dict, List

from .registry import AUG_REGISTRY
from .transforms import apply_intensity


# ---------------------------------------------------------------------
# Pool construction helpers
# ---------------------------------------------------------------------

def _collect_intensities(family_key: str) -> List[dict]:
    """
    Flatten all intensity configurations for a given family.

    Example:
        geometric -> 3 sub-augs x 3 levels = 9 configs
    """
    return [
        intensity
        for entry in AUG_REGISTRY[family_key]
        for intensity in entry["intensities"]
    ]


def _random_intensity(pool: List[dict], seed: int) -> dict:
    """
    Deterministic sampling from a pool using a local RNG.

    Why this matters:
    - ensures reproducibility per training step
    - avoids global random state pollution
    """
    rng = random.Random(seed)
    return rng.choice(pool)


def _make_policy(pool: List[dict]) -> Callable:
    """
    Build a training-time augmentation policy.

    Workflow:
    1. Use sample seed to select one deterministic intensity config
    2. Apply augmentation via apply_intensity()
    """
    def policy(image: np.ndarray, seed: int) -> np.ndarray:
        intensity = _random_intensity(pool, seed)
        return apply_intensity(image, intensity, seed)

    return policy


# ---------------------------------------------------------------------
# Policy implementations
# ---------------------------------------------------------------------

def _policy_m0(image: np.ndarray, seed: int) -> np.ndarray:
    """
    M0 - Baseline (no augmentation)

    Used as:
    - control model
    - clean reference performance
    """
    return image.copy()


# Precompute intensity pools (important for efficiency + consistency)
_GEO_POOL = _collect_intensities("geometric")
_WX_POOL  = _collect_intensities("weather")
_BN_POOL  = _collect_intensities("blur_noise")
_IL_POOL  = _collect_intensities("illumination")

# Full augmentation space (used by M5)
_ALL_POOL = _GEO_POOL + _WX_POOL + _BN_POOL + _IL_POOL


# ---------------------------------------------------------------------
# Policy factories (M1 - M5)
# ---------------------------------------------------------------------

_policy_m1 = _make_policy(_GEO_POOL)   # Geometric transformations only
_policy_m2 = _make_policy(_WX_POOL)    # Weather conditions only
_policy_m3 = _make_policy(_BN_POOL)    # Blur / noise degradation only
_policy_m4 = _make_policy(_IL_POOL)    # Illumination changes only
_policy_m5 = _make_policy(_ALL_POOL)   # Fully mixed robustness policy


# ---------------------------------------------------------------------
# Public registry
# ---------------------------------------------------------------------

POLICIES: Dict[str, dict] = {
    "M0": {
        "fn": _policy_m0,
        "name": "Baseline",
        "family": "none",
        "pool_size": 0,
    },
    "M1": {
        "fn": _policy_m1,
        "name": "Geometric",
        "family": "geometric",
        "pool_size": len(_GEO_POOL),
    },
    "M2": {
        "fn": _policy_m2,
        "name": "Weather",
        "family": "weather",
        "pool_size": len(_WX_POOL),
    },
    "M3": {
        "fn": _policy_m3,
        "name": "Blur / Noise",
        "family": "blur_noise",
        "pool_size": len(_BN_POOL),
    },
    "M4": {
        "fn": _policy_m4,
        "name": "Illumination",
        "family": "illumination",
        "pool_size": len(_IL_POOL),
    },
    "M5": {
        "fn": _policy_m5,
        "name": "Mixed Robust",
        "family": "all",
        "pool_size": len(_ALL_POOL),
    },
}


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def get_policy(model_id: str) -> Callable:
    """
    Retrieve augmentation policy function for a model.

    Parameters
    ----------
    model_id : str
        One of ["M0", "M1", "M2", "M3", "M4", "M5"]

    Returns
    -------
    Callable
        Function: (image, seed) -> augmented image
    """
    if model_id not in POLICIES:
        raise ValueError(
            f"Unknown model_id '{model_id}'. "
            f"Valid options: {list(POLICIES.keys())}"
        )
    return POLICIES[model_id]["fn"]


def describe_policies() -> None:
    """
    Print a compact summary of all augmentation policies.

    Useful for:
    - debugging training configuration
    - report generation
    - reproducibility logs
    """
    print(f"{'Model':<6} {'Name':<18} {'Family':<16} {'Pool size':>10}")
    print("-" * 56)

    for mid, info in POLICIES.items():
        print(
            f"{mid:<6} {info['name']:<18} {info['family']:<16} {info['pool_size']:>10}"
        )