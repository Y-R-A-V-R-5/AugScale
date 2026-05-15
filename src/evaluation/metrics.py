"""
src/evaluation/metrics.py
==========================
Lightweight metric aggregation layer for Phase-2 YOLOv8 evaluation.

This module collects per-condition evaluation outputs and converts them
into a structured pandas DataFrame suitable for:

- ranking computation
- plotting
- statistical analysis
- robustness scoring

Design Goal
-----------
Evaluation is executed per (model x corruption condition). This class
ensures all results are stored in a consistent schema before downstream
processing.
"""

from __future__ import annotations

from typing import List
import pandas as pd

from src.evaluation.conditions import EvalCondition


# ---------------------------------------------------------------------
# Metrics Collector
# ---------------------------------------------------------------------

class MetricsCollector:
    """
    Accumulates evaluation metrics for a single model.

    Each row corresponds to one evaluation condition:
        (clean or corrupted image configuration)

    The collector acts as an in-memory buffer before exporting to
    a pandas DataFrame.
    """

    def __init__(self, model_id: str):
        """
        Initialize collector for a specific model.

        Parameters
        ----------
        model_id : str
            Experiment identifier (M0-M5)
        """

        # Store model identity for all recorded rows
        self.model_id = model_id

        # Internal storage for metric rows
        self._rows: List[dict] = []

    # -----------------------------------------------------------------
    # Add Single Evaluation Result
    # -----------------------------------------------------------------

    def add(self, cond: EvalCondition, metrics: dict) -> None:
        """
        Add evaluation result for a single condition.

        Parameters
        ----------
        cond : EvalCondition
            Metadata describing the evaluation condition

        metrics : dict
            Dictionary containing evaluation outputs such as:
                - map50
                - precision
                - recall
                - etc.
        """

        # -------------------------------------------------------------
        # Construct structured row
        # -------------------------------------------------------------

        row = dict(

            # Model identifier (M0-M5)
            model_id=self.model_id,

            # Unique condition index
            condition_id=cond.condition_id,

            # Human-readable condition label
            label=cond.label,

            # Augmentation family (geometric/weather/etc.)
            family=cond.family_key or "clean",

            # Specific augmentation subtype
            sub_aug=cond.sub_aug_name or " - ",

            # Intensity level label (mild/moderate/strong)
            intensity_level=(
                cond.intensity["level"].capitalize()
                if cond.intensity
                else " - "
            ),

            # Random seed used for reproducibility
            seed=cond.seed,

            # Grid position in evaluation matrix
            row_idx=cond.row_idx,
            col_idx=cond.col_idx,
        )

        # Merge actual evaluation metrics
        row.update(metrics)

        # Store result
        self._rows.append(row)

    # -----------------------------------------------------------------
    # Single-row export (for incremental CSV writing)
    # -----------------------------------------------------------------

    def latest_row_as_dataframe(self, delta_map50: float) -> pd.DataFrame:
        """
        Return the most recently added condition as a one-row DataFrame
        with delta_map50 already set.

        Called by ModelEvaluator immediately after each condition so the
        row can be appended to both CSVs without waiting for all 37
        conditions to finish.

        Parameters
        ----------
        delta_map50 : float
            Degradation relative to clean baseline.
            Pass 0.0 for condition 0 (baseline is its own reference).
            Pass float("nan") if baseline is not yet known (should not
            happen in normal flow since condition 0 always runs first).

        Returns
        -------
        pd.DataFrame
            Single-row DataFrame with all columns including delta_map50.
        """
        row = dict(self._rows[-1])   # copy of the row just added
        row["delta_map50"] = delta_map50
        return pd.DataFrame([row])

    # -----------------------------------------------------------------
    # Full export to DataFrame
    # -----------------------------------------------------------------

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert all collected metrics into a pandas DataFrame.

        Note: delta_map50 is left as NaN here because the evaluator
        recomputes it from the stored baseline after the loop.

        Returns
        -------
        pd.DataFrame
            Structured evaluation table, sorted by condition_id.
        """

        df = pd.DataFrame(self._rows)

        # delta_map50 placeholder — overwritten by the evaluator
        df["delta_map50"] = float("nan")

        return (
            df.sort_values("condition_id")
              .reset_index(drop=True)
        )