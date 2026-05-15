"""
src/augmentations/__init__.py
=============================
Central export hub for all augmentation components in Phase-2.

This module ensures a single import point for:
- registry (ground-truth augmentation definitions)
- policies (training-time augmentation behavior)
- transforms (low-level deterministic image ops)
- albumentations builder (YOLOv8 training pipeline)

Design goal:
------------
Prevent import drift across:
training / evaluation / visualization / experiments
"""

from .registry import AUG_REGISTRY, FAMILY_KEYS
from .policies import get_policy, POLICIES
from .transforms import build_albumentations_transform, apply_intensity

__all__ = [
    # Core registry
    "AUG_REGISTRY",
    "FAMILY_KEYS",

    # Training policies
    "get_policy",
    "POLICIES",

    # Transform utilities
    "apply_intensity",
    "build_albumentations_transform",
]