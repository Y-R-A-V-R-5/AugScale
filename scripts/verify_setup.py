"""
scripts/verify_setup.py
========================
Smoke-test script for Phase-2 pipeline integrity.

Verifies:
- Config files exist and load correctly
- Augmentation registry consistency
- Transform functions execute safely
- 37-condition evaluation grid correctness
- Active model definitions in config
- Policy system consistency

Run before any training to avoid silent pipeline failures.
"""

from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------
# Project root setup (enable src imports from CLI)
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

# ---------------------------------------------------------------------
# Core imports
# ---------------------------------------------------------------------
from src.augmentations.registry import AUG_REGISTRY, all_sub_augs, FAMILY_KEYS
from src.augmentations.policies import describe_policies
from src.augmentations.transforms import apply_intensity
from src.evaluation.conditions import build_eval_conditions
from src.utils.io import load_yaml
from src.utils.seed import set_global_seed


# ---------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------
def check_configs() -> bool:
    """
    Ensure required YAML configs exist and are readable.
    """
    ok = True

    for cfg_path in [
        "configs/train.yaml",
        "configs/evaluation.yaml",
        "configs/augmentations.yaml",
    ]:
        p = Path(cfg_path)

        if not p.exists():
            print(f"  [FAIL] Missing config: {cfg_path}")
            ok = False
            continue

        try:
            _ = load_yaml(p)
            print(f"  [OK] {cfg_path}")
        except Exception as e:
            print(f"  [FAIL] Invalid YAML {cfg_path}: {e}")
            ok = False

    return ok


# ---------------------------------------------------------------------
# Registry validation
# ---------------------------------------------------------------------
def check_registry() -> bool:
    """
    Validate augmentation registry structure:
    - correct number of families
    - each sub-aug has exactly 3 intensity levels
    """
    subs = all_sub_augs()

    print(f"\n  Registry: {len(subs)} sub-augmentations across {len(FAMILY_KEYS)} families")

    ok = True

    for fam_key in FAMILY_KEYS:
        entries = AUG_REGISTRY[fam_key]

        for entry in entries:
            n = len(entry["intensities"])

            if n != 3:
                print(f"  [FAIL] {entry['name']} -> expected 3 intensities, got {n}")
                ok = False
            else:
                print(f"  [OK] {entry['name']}")

    return ok


# ---------------------------------------------------------------------
# Transform sanity check
# ---------------------------------------------------------------------
def check_transforms() -> bool:
    """
    Run every augmentation on a dummy image to ensure:
    - no runtime errors
    - output shape is preserved
    """
    dummy = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)

    ok = True
    total = 0

    for entry in all_sub_augs():
        for intensity in entry["intensities"]:
            total += 1
            try:
                out = apply_intensity(dummy, intensity, seed=42)

                if out.shape != dummy.shape:
                    raise ValueError(f"Shape mismatch: {out.shape}")

            except Exception as e:
                print(f"  [FAIL] {entry['name']} ({intensity['level']}): {e}")
                ok = False

    if ok:
        print(f"  [OK] All {total} transform runs passed")

    return ok


# ---------------------------------------------------------------------
# Condition grid validation
# ---------------------------------------------------------------------
def check_conditions() -> bool:
    """
    Validate that evaluation grid contains exactly 37 conditions.
    """
    conditions = build_eval_conditions()

    ok = len(conditions) == 37

    print(f"  {'[OK]' if ok else '[FAIL]'} Conditions: {len(conditions)} (expected 37)")

    return ok


# ---------------------------------------------------------------------
# Active model validation
# ---------------------------------------------------------------------
def check_active_models() -> bool:
    """
    Ensure active_models in train.yaml are defined in models section.
    """
    cfg = load_yaml("configs/train.yaml")

    all_ids = set(cfg.get("models", {}).keys())
    active = cfg.get("active_models", list(all_ids))

    ok = True

    for mid in active:
        if mid not in all_ids:
            print(f"  [FAIL] {mid} not defined in models:")
            ok = False
        else:
            print(f"  [OK] {mid}")

    return ok


# ---------------------------------------------------------------------
# Main verification pipeline
# ---------------------------------------------------------------------
def main() -> None:
    set_global_seed(42)

    print("\nPhase-2 Setup Verification")
    print("=" * 50)

    print("\n[1] Configs")
    c1 = check_configs()

    print("\n[2] Registry")
    c2 = check_registry()

    print("\n[3] Transforms")
    c3 = check_transforms()

    print("\n[4] Conditions")
    c4 = check_conditions()

    print("\n[5] Active models")
    c5 = check_active_models()

    print("\n[6] Policies")
    describe_policies()

    print("\n" + "=" * 50)

    if all([c1, c2, c3, c4, c5]):
        print("  [OK] Setup valid - ready for training pipeline")
    else:
        print("  [FAIL] Setup invalid - fix errors before running training")
        sys.exit(1)


if __name__ == "__main__":
    main()