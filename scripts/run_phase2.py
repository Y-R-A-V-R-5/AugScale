"""
scripts/run_phase2.py
======================
Master script: runs the complete Phase-2 pipeline end-to-end.

    Train M0-M5 -> Evaluate all 37 conditions -> Rank -> Generate plots

Which models are active is read from active_models in configs/train.yaml.
No model IDs are hardcoded here.

Usage
-----
# Full pipeline (all active_models)
python scripts/run_phase2.py

# Quick smoke test (2 epochs, CPU)
python scripts/run_phase2.py --smoke-test

# Skip training, just evaluate + plot from existing weights
python scripts/run_phase2.py --skip-train

# Train only specific models
python scripts/run_phase2.py --models M0 M1 M2
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

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

log = get_logger("scripts.run_phase2")

TRAIN_CFG = "configs/train.yaml"
EVAL_CFG = "configs/evaluation.yaml"
WEIGHTS_PATTERN = "experiments/{model_id}/weights/best.pt"


# ---------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------

def _active_models(cfg_path: str = TRAIN_CFG) -> List[str]:
    """
    Read active models from train config.

    Fallback:
        If active_models is not defined, use all models in config.
    """
    cfg = load_yaml(cfg_path)
    all_ids = list(cfg.get("models", {}).keys())
    return cfg.get("active_models", all_ids)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args(default_models: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase-2 full pipeline (train -> eval -> rank -> plots)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    p.add_argument(
        "--models",
        nargs="+",
        default=default_models,
        choices=default_models,
        help="Subset of models to process.",
    )

    p.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training stage (use existing weights).",
    )

    p.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip evaluation stage (use existing results).",
    )

    p.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run quick pipeline (2 epochs, CPU).",
    )

    p.add_argument(
        "--list-policies",
        action="store_true",
        help="Print augmentation policies and exit.",
    )

    return p.parse_args()


# ---------------------------------------------------------------------
# Core pipeline stages
# ---------------------------------------------------------------------

def _train(models: List[str], smoke_test: bool) -> tuple[list[str], list[str]]:
    """
    Train selected models.

    Returns:
        (success_models, failed_models)
    """
    overrides = {}

    if smoke_test:
        log.info("[SMOKE TEST] epochs=2, device=cpu, batch=4")
        overrides = {"epochs": 2, "device": "cpu", "batch_size": 4}

    ok, fail = [], []

    for mid in models:
        try:
            trainer = ModelTrainer(mid, TRAIN_CFG, overrides=overrides)
            result = trainer.run()

            log.info(
                f"[OK] {mid} | "
                f"{result.elapsed_sec / 60:.1f} min | "
                f"best={result.best_pt}"
            )
            ok.append(mid)

        except Exception as e:
            log.error(f"[FAIL] Training {mid}: {e}")
            fail.append(mid)

    return ok, fail


def _evaluate(models: List[str]) -> tuple[list[str], list[str]]:
    """
    Run 37-condition evaluation for each model.
    """
    ok, fail = [], []

    for mid in models:
        weights = Path(WEIGHTS_PATTERN.format(model_id=mid))

        if not weights.exists():
            log.warning(f"[SKIP] {mid} missing weights: {weights}")
            fail.append(mid)
            continue

        try:
            evaluator = ModelEvaluator(mid, weights, EVAL_CFG)
            evaluator.run()
            ok.append(mid)

        except Exception as e:
            log.error(f"[FAIL] Evaluation {mid}: {e}")
            fail.append(mid)

    return ok, fail


def _plots() -> None:
    """
    Generate ranking + all visualizations.
    """
    raw_csv = Path("results/raw_eval.csv")

    if not raw_csv.exists():
        log.warning("raw_eval.csv not found -> skipping plots")
        return

    df = pd.read_csv(raw_csv)
    ranking = compute_robustness_ranking()

    plot_sensitivity_heatmaps(df)
    plot_degradation_curves(df)
    plot_model_comparison(ranking)

    log.info("[OK] Plots generated")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    default_models = _active_models()
    args = parse_args(default_models)

    if args.list_policies:
        describe_policies()
        sys.exit(0)

    set_global_seed(42)

    t0 = time.time()

    log.info("=" * 60)
    log.info("Phase-2 Robustness Pipeline")
    log.info(f"Models: {args.models}")
    log.info("=" * 60)

    # -------------------------
    # Stage 1: Training
    # -------------------------
    trained_ok, trained_fail = [], []

    if args.skip_train:
        log.info("Stage 1 skipped (training)")
        trained_ok = args.models
    else:
        log.info("Stage 1: Training")
        trained_ok, trained_fail = _train(args.models, args.smoke_test)

    # -------------------------
    # Stage 2: Evaluation
    # -------------------------
    eval_ok, eval_fail = [], []

    if args.skip_eval:
        log.info("Stage 2 skipped (evaluation)")
    else:
        log.info("Stage 2: Evaluation")
        eval_ok, eval_fail = _evaluate(args.models)

    # -------------------------
    # Stage 3: Ranking + plots
    # -------------------------
    log.info("Stage 3: Ranking + Plots")

    raw_csv = Path("results/raw_eval.csv")
    if raw_csv.exists():
        compute_robustness_ranking()
        _plots()
    else:
        log.warning("No evaluation results found")

    # -------------------------
    # Summary
    # -------------------------
    elapsed = (time.time() - t0) / 60

    log.info("=" * 60)
    log.info("Pipeline Complete")
    log.info(f"Time: {elapsed:.1f} min")
    log.info(f"Trained OK: {trained_ok}")
    log.info(f"Eval OK: {eval_ok}")

    if trained_fail or eval_fail:
        log.info(f"Train failed: {trained_fail}")
        log.info(f"Eval failed: {eval_fail}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()