"""
src/augmentations/registry.py
==============================
Single source of truth for all Phase-2 augmentations.

Used by:
- training pipeline (M0-M5)
- evaluation pipeline (37-condition robustness test)

Guarantees zero parameter drift between:
augmentation.ipynb -> training -> evaluation -> ranking
"""

from __future__ import annotations
from typing import Dict, List, TypedDict, Literal

# ---------------------------------------------------------------------
# Global constants
# ---------------------------------------------------------------------

COL_SEEDS: List[int] = [42, 137, 256]
FAMILY_KEYS = ["geometric", "weather", "blur_noise", "illumination"]


# ---------------------------------------------------------------------
# Typed structure (lightweight, no dataclass overhead)
# ---------------------------------------------------------------------

Level = Literal["mild", "moderate", "strong"]

class Intensity(TypedDict, total=False):
    level: str
    aug: str


class SubAugDef(TypedDict):
    name: str
    family: str
    family_key: str
    fam_color: str
    row_labels: List[str]
    col_seeds: List[int]
    intensities: List[Intensity]


# ---------------------------------------------------------------------
# FAMILY A - Geometric
# ---------------------------------------------------------------------

_GEOMETRIC: List[SubAugDef] = [
    {
        "name": "Perspective Shift",
        "family": "A - Geometric",
        "family_key": "geometric",
        "fam_color": "#2980b9",
        "row_labels": [
            "Mild (0.02-0.04)",
            "Moderate (0.06-0.09)",
            "Strong (0.12-0.16)",
        ],
        "col_seeds": COL_SEEDS,
        "intensities": [
            {"level": "mild", "aug": "perspective", "scale": (0.02, 0.04)},
            {"level": "moderate", "aug": "perspective", "scale": (0.06, 0.09)},
            {"level": "strong", "aug": "perspective", "scale": (0.12, 0.16)},
        ],
    },
    {
        "name": "Affine Transform",
        "family": "A - Geometric",
        "family_key": "geometric",
        "fam_color": "#2980b9",
        "row_labels": [
            "Mild",
            "Moderate",
            "Strong",
        ],
        "col_seeds": COL_SEEDS,
        "intensities": [
            {
                "level": "mild",
                "aug": "affine",
                "rotate": (-5, 5),
                "scale": (0.95, 1.05),
                "translate_percent": (-0.03, 0.03),
            },
            {
                "level": "moderate",
                "aug": "affine",
                "rotate": (-12, 12),
                "scale": (0.88, 1.12),
                "translate_percent": (-0.08, 0.08),
            },
            {
                "level": "strong",
                "aug": "affine",
                "rotate": (-20, 20),
                "scale": (0.80, 1.20),
                "translate_percent": (-0.14, 0.14),
            },
        ],
    },
    {
        "name": "Horizontal Flip",
        "family": "A - Geometric",
        "family_key": "geometric",
        "fam_color": "#2980b9",
        "row_labels": ["Flip", "Flip", "Flip"],
        "col_seeds": COL_SEEDS,
        "intensities": [
            {"level": "mild", "aug": "hflip", "p": 1.0},
            {"level": "moderate", "aug": "hflip", "p": 1.0},
            {"level": "strong", "aug": "hflip", "p": 1.0},
        ],
    },
]


# ---------------------------------------------------------------------
# FAMILY B - Weather
# ---------------------------------------------------------------------

_WEATHER: List[SubAugDef] = [
    {
        "name": "Fog Simulation",
        "family": "B - Weather",
        "family_key": "weather",
        "fam_color": "#27ae60",
        "row_labels": ["Light", "Moderate", "Dense"],
        "col_seeds": COL_SEEDS,
        "intensities": [
            {"level": "light", "aug": "fog", "alpha": 0.20},
            {"level": "moderate", "aug": "fog", "alpha": 0.45},
            {"level": "dense", "aug": "fog", "alpha": 0.72},
        ],
    },
    {
        "name": "Rain Overlay",
        "family": "B - Weather",
        "family_key": "weather",
        "fam_color": "#27ae60",
        "row_labels": ["Light", "Moderate", "Heavy"],
        "col_seeds": COL_SEEDS,
        "intensities": [
            {"level": "light", "aug": "rain", "n_drops": 150, "darkness": 0.02},
            {"level": "moderate", "aug": "rain", "n_drops": 380, "darkness": 0.12},
            {"level": "heavy", "aug": "rain", "n_drops": 700, "darkness": 0.22},
        ],
    },
    {
        "name": "Haze",
        "family": "B - Weather",
        "family_key": "weather",
        "fam_color": "#27ae60",
        "row_labels": ["Light", "Moderate", "Heavy"],
        "col_seeds": COL_SEEDS,
        "intensities": [
            {"level": "light", "aug": "haze", "strength": 0.15},
            {"level": "moderate", "aug": "haze", "strength": 0.38},
            {"level": "heavy", "aug": "haze", "strength": 0.60},
        ],
    },
]


# ---------------------------------------------------------------------
# FAMILY C - Blur / Noise
# ---------------------------------------------------------------------

_BLUR_NOISE: List[SubAugDef] = [
    {
        "name": "Gaussian Blur",
        "family": "C - Blur / Noise",
        "family_key": "blur_noise",
        "fam_color": "#8e44ad",
        "row_labels": ["Low", "Medium", "High"],
        "col_seeds": COL_SEEDS,
        "intensities": [
            {"level": "low", "aug": "gaussian_blur", "blur_limit": (3, 5)},
            {"level": "medium", "aug": "gaussian_blur", "blur_limit": (9, 13)},
            {"level": "high", "aug": "gaussian_blur", "blur_limit": (19, 23)},
        ],
    },
    {
        "name": "Motion Blur",
        "family": "C - Blur / Noise",
        "family_key": "blur_noise",
        "fam_color": "#8e44ad",
        "row_labels": ["Low", "Medium", "High"],
        "col_seeds": COL_SEEDS,
        "intensities": [
            {"level": "low", "aug": "motion_blur", "blur_limit": 7},
            {"level": "medium", "aug": "motion_blur", "blur_limit": 15},
            {"level": "high", "aug": "motion_blur", "blur_limit": 25},
        ],
    },
    {
        "name": "Gaussian Noise",
        "family": "C - Blur / Noise",
        "family_key": "blur_noise",
        "fam_color": "#8e44ad",
        "row_labels": ["Low", "Medium", "High"],
        "col_seeds": COL_SEEDS,
        "intensities": [
            {"level": "low", "aug": "gaussian_noise", "std_range": (0.01, 0.03)},
            {"level": "medium", "aug": "gaussian_noise", "std_range": (0.05, 0.09)},
            {"level": "high", "aug": "gaussian_noise", "std_range": (0.12, 0.18)},
        ],
    },
]


# ---------------------------------------------------------------------
# FAMILY D - Illumination
# ---------------------------------------------------------------------

_ILLUMINATION: List[SubAugDef] = [
    {
        "name": "Brightness Shift",
        "family": "D - Illumination",
        "family_key": "illumination",
        "fam_color": "#e67e22",
        "row_labels": ["Mild", "Moderate", "Strong"],
        "col_seeds": COL_SEEDS,
        "intensities": [
            {"level": "mild", "aug": "brightness", "brightness_limit": (-0.15, 0.15)},
            {"level": "moderate", "aug": "brightness", "brightness_limit": (-0.35, 0.35)},
            {"level": "strong", "aug": "brightness", "brightness_limit": (-0.55, 0.55)},
        ],
    },
    {
        "name": "Contrast Adjustment",
        "family": "D - Illumination",
        "family_key": "illumination",
        "fam_color": "#e67e22",
        "row_labels": ["Mild", "Moderate", "Strong"],
        "col_seeds": COL_SEEDS,
        "intensities": [
            {"level": "mild", "aug": "contrast", "contrast_limit": (-0.15, 0.15)},
            {"level": "moderate", "aug": "contrast", "contrast_limit": (-0.35, 0.35)},
            {"level": "strong", "aug": "contrast", "contrast_limit": (-0.55, 0.55)},
        ],
    },
    {
        "name": "Gamma Correction",
        "family": "D - Illumination",
        "family_key": "illumination",
        "fam_color": "#e67e22",
        "row_labels": ["Mild", "Moderate", "Strong"],
        "col_seeds": COL_SEEDS,
        "intensities": [
            {"level": "mild", "aug": "gamma", "gamma_limit": (85, 100)},
            {"level": "moderate", "aug": "gamma", "gamma_limit": (40, 70)},
            {"level": "strong", "aug": "gamma", "gamma_limit": (150, 200)},
        ],
    },
]


# ---------------------------------------------------------------------
# Master registry
# ---------------------------------------------------------------------

AUG_REGISTRY: Dict[str, List[SubAugDef]] = {
    "geometric": _GEOMETRIC,
    "weather": _WEATHER,
    "blur_noise": _BLUR_NOISE,
    "illumination": _ILLUMINATION,
}


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def all_sub_augs() -> List[SubAugDef]:
    return [x for group in AUG_REGISTRY.values() for x in group]


def get_sub_aug(name: str) -> SubAugDef:
    for item in all_sub_augs():
        if item["name"] == name:
            return item
    raise KeyError(f"Unknown augmentation: {name}")


def validate_registry() -> None:
    """
    Basic safety check to prevent silent drift.
    Call this at startup in training/eval.
    """
    for fam, items in AUG_REGISTRY.items():
        for item in items:
            assert "name" in item
            assert "intensities" in item
            assert len(item["intensities"]) == 3, f"{fam}:{item['name']}"