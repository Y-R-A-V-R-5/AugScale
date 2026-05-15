"""
src/evaluation/ranking.py
==========================
Robustness ranking computation for Phase-2 YOLOv8 experiments.

This module converts raw evaluation metrics into a single robustness
score per model and generates the final experiment ranking table.

Purpose
-------
Different augmentation policies produce different robustness behaviors.

A single scalar robustness score is computed to:
- compare models fairly
- summarize robustness performance
- identify the most stable augmentation strategy
- simplify experimental analysis

Input
-----
results/raw_eval.csv

Expected columns:
- model_id
- condition_id
- row_idx
- map50
- delta_map50

Output
------
results/robustness_ranking.csv

Generated Metrics
-----------------
For each model:

1. clean_map50
    mAP@50 under clean validation conditions

2. mean_map50_augmented
    Average mAP@50 across all augmented conditions

3. worst_map50
    Lowest mAP@50 under corruption

4. mean_delta_map50
    Average degradation relative to clean baseline

5. mean_degradation_slope
    Linear degradation trend across severity levels

6. robustness_score
    Weighted aggregate robustness metric

Scoring Formula
----------------
Weights are loaded from configs/evaluation.yaml.

The final score is:

    score =
          w_map   * mean_map50
        + w_worst * worst_map50
        + w_delta * mean_delta_map50
        + w_slope * degradation_slope

Interpretation
---------------
Higher score:
    More robust model

Lower score:
    Less stable under corruption

Important
----------
For delta and slope:
- values are typically negative
- less negative values indicate better robustness
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pathlib import Path

from src.utils.io import load_yaml, ensure_dir
from src.utils.logger import get_logger


# ---------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------

log = get_logger(__name__)


# ---------------------------------------------------------------------
# Robustness Ranking
# ---------------------------------------------------------------------

def compute_robustness_ranking(
    raw_csv: str | Path = "results/raw_eval.csv",
    eval_cfg: str | Path = "configs/evaluation.yaml",
    out_csv: str | Path = "results/robustness_ranking.csv",
) -> pd.DataFrame:
    """
    Compute robustness ranking for all experiment models.

    Parameters
    ----------
    raw_csv : str | Path
        Path to raw evaluation CSV

    eval_cfg : str | Path
        Path to evaluation configuration YAML

    out_csv : str | Path
        Output CSV path for ranking table

    Returns
    -------
    pd.DataFrame
        Ranking table sorted by robustness score
        (highest score = rank 1)

    Workflow
    --------
    1. Load evaluation metrics
    2. Group rows by model
    3. Compute robustness statistics
    4. Compute weighted robustness score
    5. Rank models
    6. Save ranking CSV
    """

    # -------------------------------------------------------------
    # Load Evaluation Configuration
    # -------------------------------------------------------------

    cfg = load_yaml(eval_cfg)

    # Weight coefficients used in final score calculation
    weights = cfg["ranking_weights"]

    # -------------------------------------------------------------
    # Load Raw Evaluation Data
    # -------------------------------------------------------------

    df = pd.read_csv(raw_csv)

    # Storage for final ranking rows
    records = []

    # -------------------------------------------------------------
    # Process Each Model Independently
    # -------------------------------------------------------------

    for model_id, grp in df.groupby("model_id"):

        # ---------------------------------------------------------
        # Split Clean vs Augmented Conditions
        # ---------------------------------------------------------

        # condition_id == 0 represents the clean baseline.
        clean = grp[grp["condition_id"] == 0]

        # condition_id > 0 represents corrupted/augmented inputs.
        augmented = grp[grp["condition_id"] > 0]

        # ---------------------------------------------------------
        # Mean Robustness Performance
        # ---------------------------------------------------------

        # Average performance across all corruption conditions.
        mean_map50 = augmented["map50"].mean()

        # ---------------------------------------------------------
        # Worst-Case Robustness
        # ---------------------------------------------------------

        # Lowest observed mAP under corruption.
        #
        # This measures how badly the model can fail.
        worst_map50 = augmented["map50"].min()

        # ---------------------------------------------------------
        # Average Performance Drop
        # ---------------------------------------------------------

        # Difference relative to clean performance.
        #
        # Typically negative:
        # smaller drop = better robustness.
        mean_delta = augmented["delta_map50"].mean()

        # ---------------------------------------------------------
        # Degradation Slope
        # ---------------------------------------------------------

        # Measures how rapidly performance declines
        # as corruption severity increases.
        #
        # A flatter slope is better.
        #
        # Linear regression:
        #   y = mx + b
        #
        # where:
        #   x = severity level
        #   y = map50
        #
        # We only need slope "m".
        if len(augmented) >= 2:

            slope = float(
                np.polyfit(
                    augmented["row_idx"],
                    augmented["map50"],
                    1,
                )[0]
            )

        else:
            # Not enough points for regression
            slope = 0.0

        # ---------------------------------------------------------
        # Weighted Robustness Score
        # ---------------------------------------------------------

        # Final scalar robustness metric.
        #
        # Weight meanings:
        #
        # mean_map50:
        #     Overall corruption performance
        #
        # worst_map50:
        #     Worst-case resilience
        #
        # mean_delta_map50:
        #     Stability relative to clean baseline
        #
        # degradation_slope:
        #     Resistance to increasing severity
        score = (
            weights["mean_map50"] * mean_map50
            + weights["worst_map50"] * worst_map50
            + weights["mean_delta_map50"] * mean_delta
            + weights["mean_degradation_slope"] * slope
        )

        # ---------------------------------------------------------
        # Store Model Summary
        # ---------------------------------------------------------

        records.append(
            dict(

                # Experiment identifier
                model_id=model_id,

                # Clean-condition performance
                clean_map50=(
                    float(clean["map50"].values[0])
                    if len(clean)
                    else float("nan")
                ),

                # Average corrupted performance
                mean_map50_augmented=round(
                    mean_map50,
                    4,
                ),

                # Worst observed corrupted performance
                worst_map50=round(
                    worst_map50,
                    4,
                ),

                # Mean degradation relative to clean
                mean_delta_map50=round(
                    mean_delta,
                    4,
                ),

                # Regression slope across corruption levels
                mean_degradation_slope=round(
                    slope,
                    5,
                ),

                # Final weighted robustness score
                robustness_score=round(
                    score,
                    4,
                ),
            )
        )

    # -------------------------------------------------------------
    # Build Ranking Table
    # -------------------------------------------------------------

    ranking = (
        pd.DataFrame(records)

        # Highest robustness first
        .sort_values(
            "robustness_score",
            ascending=False,
        )

        # Clean sequential indexing
        .reset_index(drop=True)
    )

    # -------------------------------------------------------------
    # Insert Explicit Rank Column
    # -------------------------------------------------------------

    ranking.insert(
        0,
        "rank",
        ranking.index + 1,
    )

    # -------------------------------------------------------------
    # Ensure Output Directory Exists
    # -------------------------------------------------------------

    ensure_dir(Path(out_csv).parent)

    # -------------------------------------------------------------
    # Save Ranking CSV
    # -------------------------------------------------------------

    ranking.to_csv(
        out_csv,
        index=False,
    )

    log.info(
        f"Robustness ranking saved -> {out_csv}"
    )

    # -------------------------------------------------------------
    # Pretty Console Summary
    # -------------------------------------------------------------

    print("\n" + "=" * 72)

    print(
        "  Phase-2 Robustness Ranking".center(72)
    )

    print("=" * 72)

    print(
        ranking.to_string(index=False)
    )

    print("=" * 72 + "\n")

    return ranking