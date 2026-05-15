"""
src/evaluation/conditions.py
============================
Defines the full evaluation protocol for Phase-2 YOLOv8 robustness testing.

This module constructs the complete set of evaluation conditions used
in the robustness benchmark.

Design Overview
---------------
We evaluate each model under:

1. Clean baseline condition (no corruption)
2. 36 corrupted conditions structured as:
       4 augmentation families
     x 3 sub-augmentations per family
     x 3 intensity levels
     x 3 random seeds

Total:
    1 + 36 = 37 conditions

Each condition is fully specified by an EvalCondition object,
which contains all metadata required for:

- deterministic corruption
- reproducible evaluation
- structured logging
- downstream ranking/plotting
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from src.augmentations.registry import AUG_REGISTRY, COL_SEEDS
from src.utils.io import load_yaml


# ---------------------------------------------------------------------
# Evaluation Condition Definition
# ---------------------------------------------------------------------

@dataclass
class EvalCondition:
    """
    Represents a single evaluation scenario.

    Each condition fully defines:
    - augmentation family
    - sub augmentation type
    - intensity level
    - random seed
    - grid position (row/col)
    """

    condition_id: int

    # Human-readable description for logs/plots
    label: str

    # High-level augmentation family
    # (geometric / weather / blur_noise / illumination)
    family_key: Optional[str]

    # Specific augmentation within family
    sub_aug_name: Optional[str]

    # Intensity configuration dictionary:
    # e.g. {"level": "moderate", "value": 0.4}
    intensity: Optional[dict]

    # Seed used for deterministic corruption
    seed: int

    # Position in 3x3 evaluation grid
    row_idx: int   # intensity index (0=mild,1=moderate,2=strong)
    col_idx: int   # seed index (0..2)


# ---------------------------------------------------------------------
# Condition Builder
# ---------------------------------------------------------------------

def build_eval_conditions(
    eval_cfg_path: str = "configs/evaluation.yaml",
) -> List[EvalCondition]:
    """
    Build the complete set of 37 evaluation conditions.

    Returns
    -------
    List[EvalCondition]
        Ordered list of all evaluation scenarios
    """

    # Load config (currently not heavily used, reserved for extensibility)
    cfg = load_yaml(eval_cfg_path)

    conditions: List[EvalCondition] = []

    # -------------------------------------------------------------
    # Condition 0: Clean Baseline
    # -------------------------------------------------------------

    conditions.append(
        EvalCondition(
            condition_id=0,
            label="Clean Baseline",
            family_key=None,
            sub_aug_name=None,
            intensity=None,

            # Use first seed for consistency
            seed=COL_SEEDS[0],

            row_idx=0,
            col_idx=0,
        )
    )

    # -------------------------------------------------------------
    # Corrupted Conditions (1-36)
    # -------------------------------------------------------------

    condition_id = 1

    # Fixed family order ensures reproducibility
    family_order = [
        "geometric",
        "weather",
        "blur_noise",
        "illumination",
    ]

    # Each family contributes a full 3x3 grid
    for family_key in family_order:

        # Each family has exactly 3 sub-augmentations
        sub_aug_defs = AUG_REGISTRY[family_key]

        for row_idx in range(3):

            # Select sub-augmentation for this intensity level
            sub_def = sub_aug_defs[row_idx]

            # Intensity configuration for this level
            intensity = sub_def["intensities"][row_idx]

            for col_idx, seed in enumerate(COL_SEEDS):

                # Human-readable label used in logs and plots
                label = (
                    f"{sub_def['name']} / "
                    f"{intensity['level'].capitalize()} / "
                    f"seed={seed}"
                )

                conditions.append(
                    EvalCondition(
                        condition_id=condition_id,
                        label=label,
                        family_key=family_key,
                        sub_aug_name=sub_def["name"],
                        intensity=intensity,
                        seed=seed,
                        row_idx=row_idx,
                        col_idx=col_idx,
                    )
                )

                condition_id += 1

    # -------------------------------------------------------------
    # Safety Check
    # -------------------------------------------------------------

    assert len(conditions) == 37, (
        f"Expected 37 conditions, got {len(conditions)}"
    )

    return conditions


# ---------------------------------------------------------------------
# Debug Utility
# ---------------------------------------------------------------------

def describe_conditions(conditions: List[EvalCondition]) -> None:
    """
    Print a compact human-readable table of all evaluation conditions.

    Useful for debugging and verifying evaluation setup.
    """

    print(
        f"\n{'ID':>4}  {'Family':<14}  "
        f"{'Sub-Augmentation':<32}  {'Level':<12}  {'Seed':>6}"
    )

    print("-" * 76)

    for c in conditions:

        fam = c.family_key or " - "
        sub = c.sub_aug_name or "Clean"

        level = (
            c.intensity["level"].capitalize()
            if c.intensity
            else " - "
        )

        print(
            f"{c.condition_id:>4}  "
            f"{fam:<14}  "
            f"{sub:<32}  "
            f"{level:<12}  "
            f"{c.seed:>6}"
        )