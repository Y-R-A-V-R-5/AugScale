"""
scripts/evaluate.py
====================
CLI entry point for Phase-2 robustness evaluation.

Runs:
- 37-condition evaluation per model
- aggregation into raw_eval.csv
- robustness ranking
- optional visualization pipeline

Supports:
- single model evaluation
- full model sweep (M0-M5)
- ranking-only mode (post-processing)
- full plotting pipeline

Output layout
-------------
All outputs land under experiments/<model_id>/eval/ and results/:

    experiments/
        <model_id>/
            eval/
                raw_eval.csv              <- per-model metrics
                cond_00/                  <- YOLO val output per condition
                cond_01/
                ...

    results/
        raw_eval.csv                      <- aggregated across all models
        robustness_ranking.csv
        heatmaps/
        degradation_curves/
        model_comparison.png
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------
# Project bootstrap — ensure imports resolve from project root
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Change working directory to project root so all relative paths
# (configs/, experiments/, results/) resolve correctly regardless of
# where the user invokes the script from.
import os
os.chdir(PROJECT_ROOT)

from src.evaluation.evaluator import ModelEvaluator
from src.evaluation.ranking import compute_robustness_ranking
from src.evaluation.plots import (
    plot_sensitivity_heatmaps,
    plot_degradation_curves,
    plot_model_comparison,
)

from src.utils.io import load_yaml
from src.utils.seed import set_global_seed
from src.utils.logger import get_logger

log = get_logger("scripts.evaluate")

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

WEIGHTS_PATTERN = "experiments/{model_id}/weights/best.pt"
TRAIN_CFG       = "configs/train.yaml"
EVAL_CFG        = "configs/evaluation.yaml"

# Global aggregated CSV (all models)
RESULTS_DIR = PROJECT_ROOT / "results"
RAW_CSV     = RESULTS_DIR / "raw_eval.csv"


# ---------------------------------------------------------------------
# Model discovery
# ---------------------------------------------------------------------

def _all_models(cfg_path: str = TRAIN_CFG) -> list[str]:
    """Load model IDs from training config."""
    cfg = load_yaml(cfg_path)
    return list(cfg.get("models", {}).keys())


ALL_MODELS = _all_models()


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase-2 robustness evaluation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--model", "-m",
        required=True,
        choices=ALL_MODELS + ["all"],
        help="Model ID or 'all'",
    )

    parser.add_argument(
        "--cfg",
        default=EVAL_CFG,
        help="Evaluation config path",
    )

    parser.add_argument(
        "--weights",
        default=None,
        help="Override weights path (single model only)",
    )

    parser.add_argument(
        "--plots",
        action="store_true",
        help="Generate all plots after evaluation",
    )

    parser.add_argument(
        "--ranking-only",
        action="store_true",
        help="Skip evaluation, recompute ranking from existing results",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def resolve_weights(model_id: str, override: str | None) -> Path:
    """
    Resolve model weights path.

    Priority:
    1. CLI override (if provided and single model)
    2. default experiment path
    """
    if override:
        return Path(override)
    return Path(WEIGHTS_PATTERN.format(model_id=model_id))


def evaluate_model(model_id: str, args: argparse.Namespace) -> pd.DataFrame:
    """
    Run full 37-condition evaluation for a single model.
    """
    weights = resolve_weights(model_id, args.weights)

    log.info("=" * 70)
    log.info(f"Evaluating {model_id} | weights: {weights}")
    log.info("=" * 70)

    evaluator = ModelEvaluator(
        model_id=model_id,
        weights_pt=weights,
        cfg_path=args.cfg,
        train_cfg=TRAIN_CFG,
    )

    df = evaluator.run()

    log.info(f"[OK] {model_id} completed | rows={len(df)}\n")
    return df


# ---------------------------------------------------------------------
# Plot pipeline
# ---------------------------------------------------------------------

def generate_plots(raw_csv: Path = RAW_CSV) -> None:
    """
    Full visualization pipeline (requires results/raw_eval.csv).
    Plots are saved under results/.
    """
    if not raw_csv.exists():
        log.error(f"Missing {raw_csv}. Run evaluation first.")
        return

    eval_cfg = load_yaml(EVAL_CFG)
    out      = eval_cfg["output"]

    df      = pd.read_csv(raw_csv)
    ranking = compute_robustness_ranking(
        raw_csv  = raw_csv,
        eval_cfg = EVAL_CFG,
        out_csv  = out["ranking_csv"],
    )

    log.info("Generating sensitivity heatmaps...")
    plot_sensitivity_heatmaps(df, out_dir=out["heatmap_dir"])

    log.info("Generating degradation curves...")
    plot_degradation_curves(df, out_dir=out["curve_dir"])

    log.info("Generating model comparison...")
    plot_model_comparison(ranking, out_path=out.get("model_comparison_png", "results/model_comparison.png"))

    log.info("Plot generation complete.")


# ---------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    set_global_seed(42)

    # -------------------------------------------------------------
    # Ranking-only mode
    # -------------------------------------------------------------
    if args.ranking_only:
        log.info("Recomputing robustness ranking only...")
        eval_cfg = load_yaml(EVAL_CFG)
        compute_robustness_ranking(
            raw_csv  = RAW_CSV,
            eval_cfg = EVAL_CFG,
            out_csv  = eval_cfg["output"]["ranking_csv"],
        )

        if args.plots:
            generate_plots()
        return

    # -------------------------------------------------------------
    # Evaluation mode
    # -------------------------------------------------------------
    models = ALL_MODELS if args.model == "all" else [args.model]

    log.info(f"Phase-2: evaluation started | models: {models}")

    start_time = time.time()
    failed: list[str] = []

    for mid in models:
        try:
            evaluate_model(mid, args)
        except Exception as e:
            log.error(f"[FAIL] {mid}: {e}")
            failed.append(mid)

    elapsed = (time.time() - start_time) / 60

    log.info(
        f"Evaluation finished | "
        f"succeeded={len(models) - len(failed)} "
        f"failed={len(failed)} "
        f"time={elapsed:.1f} min"
    )

    # -------------------------------------------------------------
    # Post-processing
    # -------------------------------------------------------------
    if not failed:
        log.info("Computing robustness ranking...")
        eval_cfg = load_yaml(EVAL_CFG)
        compute_robustness_ranking(
            raw_csv  = RAW_CSV,
            eval_cfg = EVAL_CFG,
            out_csv  = eval_cfg["output"]["ranking_csv"],
        )

    if args.plots:
        generate_plots()

    if failed:
        log.error(f"Failed models: {failed}")


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------

if __name__ == "__main__":
    main()
