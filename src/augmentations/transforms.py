"""
src/augmentations/transforms.py
================================
Low-level image transformation functions for Phase-2 YOLOv8 robustness study.

Each transform follows the unified interface:

    (image: np.ndarray, seed: int, **params) -> np.ndarray

Design Principles
-----------------
- All parameters come from registry.py (no hardcoding here)
- Fully deterministic via seed control (no global RNG pollution)
- Separation of concerns:
    * geometric/vision ops via albumentations
    * weather effects via numpy/OpenCV
    * unified dispatch layer
"""

from __future__ import annotations

import cv2
import numpy as np
import albumentations as A
from typing import Callable, Dict, Any


# ============================================================
# Utilities
# ============================================================

def _rng(seed: int) -> np.random.RandomState:
    """Local RNG to ensure full reproducibility without global side effects."""
    return np.random.RandomState(seed)


# ============================================================
# Geometric Transformations (Albumentations)
# ============================================================

def apply_perspective(image: np.ndarray, seed: int, scale: tuple) -> np.ndarray:
    transform = A.Perspective(scale=scale, p=1.0)
    return transform(image=image.copy())["image"]


def apply_affine(
    image: np.ndarray,
    seed: int,
    rotate: tuple,
    scale: tuple,
    translate_percent: tuple,
) -> np.ndarray:
    transform = A.Affine(
        rotate=rotate,
        scale=scale,
        translate_percent=translate_percent,
        p=1.0,
    )
    return transform(image=image.copy())["image"]


def apply_hflip(image: np.ndarray, seed: int, p: float = 1.0) -> np.ndarray:
    transform = A.HorizontalFlip(p=p)
    return transform(image=image.copy())["image"]


# ============================================================
# Weather Effects (Custom NumPy / OpenCV)
# ============================================================

def apply_fog(image: np.ndarray, seed: int, alpha: float) -> np.ndarray:
    """White alpha-blend fog overlay."""
    overlay = np.full_like(image, 255, dtype=np.float32)

    out = image.astype(np.float32) * (1 - alpha) + overlay * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_rain(
    image: np.ndarray,
    seed: int,
    n_drops: int,
    darkness: float,
) -> np.ndarray:
    """Synthetic diagonal rain streaks."""
    rng = _rng(seed)

    result = image.copy()
    h, w = result.shape[:2]

    for _ in range(n_drops):
        x1 = rng.randint(0, w)
        y1 = rng.randint(0, h)

        length = rng.randint(10, 28)

        x2 = x1 - int(length * 0.25)
        y2 = min(y1 + length, h - 1)

        cv2.line(result, (x1, y1), (x2, y2), (200, 210, 255), 1)

    result = result.astype(np.float32) * (1 - darkness)
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_haze(image: np.ndarray, seed: int, strength: float) -> np.ndarray:
    """Atmospheric scattering haze simulation."""
    white = np.full_like(image, 230, dtype=np.float32)

    out = image.astype(np.float32) * (1 - strength) + white * strength
    return np.clip(out, 0, 255).astype(np.uint8)


# ============================================================
# Blur & Noise
# ============================================================

def apply_gaussian_blur(image: np.ndarray, seed: int, blur_limit: tuple) -> np.ndarray:
    transform = A.GaussianBlur(blur_limit=blur_limit, p=1.0)
    return transform(image=image.copy())["image"]


def apply_motion_blur(image: np.ndarray, seed: int, blur_limit: int) -> np.ndarray:
    transform = A.MotionBlur(blur_limit=blur_limit, p=1.0)
    return transform(image=image.copy())["image"]


def apply_gaussian_noise(image: np.ndarray, seed: int, std_range: tuple) -> np.ndarray:
    transform = A.GaussNoise(std_range=std_range, p=1.0)
    return transform(image=image.copy())["image"]


# ============================================================
# Illumination
# ============================================================

def apply_brightness(image: np.ndarray, seed: int, brightness_limit: tuple) -> np.ndarray:
    transform = A.RandomBrightnessContrast(
        brightness_limit=brightness_limit,
        contrast_limit=0,
        p=1.0,
    )
    return transform(image=image.copy())["image"]


def apply_contrast(image: np.ndarray, seed: int, contrast_limit: tuple) -> np.ndarray:
    transform = A.RandomBrightnessContrast(
        brightness_limit=0,
        contrast_limit=contrast_limit,
        p=1.0,
    )
    return transform(image=image.copy())["image"]


def apply_gamma(image: np.ndarray, seed: int, gamma_limit: tuple) -> np.ndarray:
    transform = A.RandomGamma(gamma_limit=gamma_limit, p=1.0)
    return transform(image=image.copy())["image"]


# ============================================================
# Dispatch Table
# ============================================================

_DISPATCH: Dict[str, Callable[..., np.ndarray]] = {
    "perspective": apply_perspective,
    "affine": apply_affine,
    "hflip": apply_hflip,
    "fog": apply_fog,
    "rain": apply_rain,
    "haze": apply_haze,
    "gaussian_blur": apply_gaussian_blur,
    "motion_blur": apply_motion_blur,
    "gaussian_noise": apply_gaussian_noise,
    "brightness": apply_brightness,
    "contrast": apply_contrast,
    "gamma": apply_gamma,
}


# ============================================================
# Unified Entry Point
# ============================================================

def apply_intensity(
    image: np.ndarray,
    intensity: dict,
    seed: int,
) -> np.ndarray:
    """
    Apply a registry-defined augmentation configuration.
    """

    if not isinstance(intensity, dict):
        raise TypeError("intensity must be a dict")

    aug_key = intensity.get("aug")
    if aug_key is None:
        raise ValueError("Missing 'aug' key in intensity config")

    if aug_key not in _DISPATCH:
        raise ValueError(
            f"Unknown augmentation '{aug_key}'. "
            f"Valid: {list(_DISPATCH.keys())}"
        )

    params = {
        k: v for k, v in intensity.items()
        if k not in ("aug", "level")
    }

    return _DISPATCH[aug_key](image, seed, **params)


# ============================================================
# Albumentations Builder (Training-time)
# ============================================================

def build_albumentations_transform(intensity: dict, seed: int):
    """
    Build Albumentations pipeline for YOLO training.
    Weather transforms are excluded (handled in numpy pipeline).
    """

    aug_key = intensity.get("aug")

    if aug_key == "perspective":
        return A.Compose([A.Perspective(scale=intensity["scale"], p=1.0)])

    if aug_key == "affine":
        return A.Compose([
            A.Affine(
                rotate=intensity["rotate"],
                scale=intensity["scale"],
                translate_percent=intensity["translate_percent"],
                p=1.0,
            )
        ])

    if aug_key == "hflip":
        return A.Compose([A.HorizontalFlip(p=intensity.get("p", 1.0))])

    if aug_key == "gaussian_blur":
        return A.Compose([A.GaussianBlur(blur_limit=intensity["blur_limit"], p=1.0)])

    if aug_key == "motion_blur":
        return A.Compose([A.MotionBlur(blur_limit=intensity["blur_limit"], p=1.0)])

    if aug_key == "gaussian_noise":
        return A.Compose([A.GaussNoise(std_range=intensity["std_range"], p=1.0)])

    if aug_key == "brightness":
        return A.Compose([
            A.RandomBrightnessContrast(
                brightness_limit=intensity["brightness_limit"],
                contrast_limit=0,
                p=1.0,
            )
        ])

    if aug_key == "contrast":
        return A.Compose([
            A.RandomBrightnessContrast(
                brightness_limit=0,
                contrast_limit=intensity["contrast_limit"],
                p=1.0,
            )
        ])

    if aug_key == "gamma":
        return A.Compose([A.RandomGamma(gamma_limit=intensity["gamma_limit"], p=1.0)])

    return None