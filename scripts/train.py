"""
scripts/train.py
=================
CLI entry point: train one or all Phase-2 models.

Which models exist is defined in configs/train.yaml under `models:`.
Which models are trained by default ("all") is controlled by `active_models:`.
No model IDs are hardcoded here.

Usage
-----
# Train the active_models list from configs/train.yaml
python scripts/train.py --model all

# Train a single model
python scripts/train.py --model M1

# Train with hyperparameter overrides
python scripts/train.py --model M2 --epochs 30 --device cpu

# Show all available model IDs
python scripts/train.py --list-policies
"""

# Master pipeline for Phase-2:
# Train -> Evaluate (37 conditions) -> Rank -> Plot results

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------
# Project setup (ensure src/ imports work from CLI)
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------
# Core modules
# ---------------------------------------------------------------------
from src.training.trainer import ModelTrainer
from src.evaluation.evaluator import ModelEvaluator
from src.evaluation.ranking import compute_robustness_ranking
from src.evaluation.plots import (
    plot_sensitivity_heatmaps,
    plot_degradation_curves,
    plot_model_comparison,
)
from src.augmentations.policies import describe_policies
from src.utils.io import load_yaml
from src.utils.seed import set_global_seed
from src.utils.logger import get_logger

import pandas as pd

# ---------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------
log = get_logger("scripts.run_phase2")

# ---------------------------------------------------------------------
# Config paths
# ---------------------------------------------------------------------
TRAIN_CFG = "configs/train.yaml"
EVAL_CFG  = "configs/evaluation.yaml"

WEIGHTS_PATTERN = "experiments/{model_id}/weights/best.pt"


# ---------------------------------------------------------------------
# Utility: active models from config
# ---------------------------------------------------------------------
def _active_models(cfg_path: str = TRAIN_CFG):
    cfg = load_yaml(cfg_path)
    all_ids = list(cfg.get("models", {}).keys())
    return cfg.get("active_models", all_ids)


# ---------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------
def parse_args(default_models) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase-2 full pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument(
        "--models",
        nargs="+",
        default=default_models,
        choices=default_models,
        help="Subset of models to run",
    )

    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-eval", action="store_true")

    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--list-policies", action="store_true")

    return p.parse_args()


# ---------------------------------------------------------------------
# Model evaluation wrapper
# ---------------------------------------------------------------------
def evaluate_model(model_id: str, cfg_path: str):
    weights = Path(WEIGHTS_PATTERN.format(model_id=model_id))

    if not weights.exists():
        raise FileNotFoundError(f"Missing weights: {weights}")

    ev = ModelEvaluator(model_id, weights, cfg_path)
    return ev.run()


# ---------------------------------------------------------------------
# Plotting pipeline
# ---------------------------------------------------------------------
def generate_plots():
    raw_csv = Path("results/raw_eval.csv")

    if not raw_csv.exists():
        log.warning("raw_eval.csv not found; skipping plots")
        return

    df = pd.read_csv(raw_csv)
    ranking = compute_robustness_ranking()

    plot_sensitivity_heatmaps(df)
    plot_degradation_curves(df)
    plot_model_comparison(ranking)


# ---------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------
def main() -> None:
    default_models = _active_models()
    args = parse_args(default_models)

    # Optional: print augmentation policies
    if args.list_policies:
        describe_policies()
        return

    set_global_seed(42)

    log.info(f"Running Phase-2 for models: {args.models}")
    t0 = time.time()

    # -------------------------
    # Stage 1: Training skipped/assumed external
    # -------------------------
    if not args.skip_train:
        log.info("Stage 1: Training skipped in this pipeline version")

    # -------------------------
    # Stage 2: Evaluation
    # -------------------------
    if not args.skip_eval:
        log.info("Stage 2: Evaluation (37 conditions)")

        failed = []
        for mid in args.models:
            try:
                evaluate_model(mid, EVAL_CFG)
                log.info(f"[OK] {mid}")
            except Exception as e:
                log.error(f"[FAIL] {mid}: {e}")
                failed.append(mid)

    # -------------------------
    # Stage 3: Ranking + plots
    # -------------------------
    log.info("Stage 3: Ranking + plots")
    compute_robustness_ranking()
    generate_plots()

    log.info(f"Done in {(time.time() - t0)/60:.1f} min")


if __name__ == "__main__":
    main()