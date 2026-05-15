"""
src/evaluation/plots.py
========================
Visualization utilities for Phase-2 YOLOv8 robustness experiments.

This module generates all evaluation figures from:
    results/raw_eval.csv

Outputs
-------
All figures are saved under results/:

1. heatmaps/
   - 3x3 mAP@0.5 sensitivity grids per model per augmentation family

2. degradation_curves/
   - mAP@0.5 vs intensity level curves per family

3. model_comparison.png
   - Overall robustness comparison across M0-M5

Design Goals
------------
These plots are designed to:
- visualize robustness degradation patterns
- compare augmentation families
- highlight model stability under increasing severity
- support thesis/report figures
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from pathlib import Path
from tqdm import tqdm

from src.augmentations.registry import AUG_REGISTRY, FAMILY_KEYS
from src.utils.io import load_yaml, ensure_dir
from src.utils.logger import get_logger


# ---------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------

log = get_logger(__name__)


# ---------------------------------------------------------------------
# Visual Style Configuration
# ---------------------------------------------------------------------

# Human-readable labels for augmentation families
FAMILY_LABELS = {
    "geometric": "A  -  Geometric",
    "weather": "B  -  Weather",
    "blur_noise": "C  -  Blur / Noise",
    "illumination": "D  -  Illumination",
}

# Color palette per augmentation family
FAMILY_COLORS = {
    "geometric": "#2980b9",
    "weather": "#27ae60",
    "blur_noise": "#8e44ad",
    "illumination": "#e67e22",
}

# Color mapping per trained model
MODEL_COLORS = {
    "M0": "#95a5a6",
    "M1": "#2980b9",
    "M2": "#27ae60",
    "M3": "#8e44ad",
    "M4": "#e67e22",
    "M5": "#e74c3c",
}


# ---------------------------------------------------------------------
# Figure Helper
# ---------------------------------------------------------------------

def _dark_fig(**kwargs) -> plt.Figure:
    """
    Create a dark-themed matplotlib figure.

    Returns
    -------
    Figure
        Matplotlib figure with dark background
    """

    fig = plt.figure(**kwargs)

    # Dark theme background for publication-style visuals
    fig.patch.set_facecolor("#0d0d1a")

    return fig


# ---------------------------------------------------------------------
# 1. Sensitivity Heatmaps
# ---------------------------------------------------------------------

def plot_sensitivity_heatmaps(
    df: pd.DataFrame,
    out_dir: str | Path = "results/heatmaps",
) -> None:
    """
    Generate 3x3 heatmaps of mAP@0.5 per model and augmentation family.

    Grid structure:
        rows -> intensity levels (0,1,2)
        cols -> random seeds (3 seeds)

    Each cell represents:
        mAP@0.5 under a specific corruption configuration

    Parameters
    ----------
    df : pd.DataFrame
        Evaluation results dataframe

    out_dir : str | Path
        Output directory for heatmaps
    """

    # -------------------------------------------------------------
    # Prepare Output Directory
    # -------------------------------------------------------------

    out_dir = Path(out_dir)
    ensure_dir(out_dir)

    models = sorted(df["model_id"].unique())

    # Total plotting tasks for progress tracking
    total_jobs = len(models) * len(FAMILY_KEYS)

    # -------------------------------------------------------------
    # Iterate Over Models and Families
    # -------------------------------------------------------------

    with tqdm(total=total_jobs, desc="Heatmaps") as pbar:

        for model_id in models:

            # Filter model-specific data once
            mdf = df[df["model_id"] == model_id]

            for fam_key in FAMILY_KEYS:

                # Filter family-specific data
                fdf = mdf[mdf["family"] == fam_key]

                if fdf.empty:
                    pbar.update(1)
                    continue

                # -------------------------------------------------
                # Build 3x3 Heatmap Grid
                # -------------------------------------------------

                grid = np.full((3, 3), np.nan)

                for _, row in fdf.iterrows():

                    r, c = int(row["row_idx"]), int(row["col_idx"])

                    if 0 <= r < 3 and 0 <= c < 3:
                        grid[r, c] = row["map50"]

                # Optional registry names (for debugging/extension)
                sub_names = [
                    e["name"]
                    for e in AUG_REGISTRY[fam_key]
                ]

                # Axis labels
                row_labels = ["L0", "L1", "L2"]
                col_labels = ["seed=42", "seed=137", "seed=256"]

                # -------------------------------------------------
                # Plot Heatmap
                # -------------------------------------------------

                fig, ax = plt.subplots(figsize=(8, 5))

                fig.patch.set_facecolor("#0d0d1a")
                ax.set_facecolor("#0d0d1a")

                sns.heatmap(
                    grid,
                    ax=ax,
                    vmin=0,
                    vmax=1,
                    cmap="RdYlGn",
                    annot=True,
                    fmt=".3f",
                    linewidths=0.5,
                    linecolor="#222244",
                    xticklabels=col_labels,
                    yticklabels=row_labels,
                    cbar_kws={"label": "mAP@0.5"},
                )

                # Title styling
                ax.set_title(
                    f"{model_id}  |  {FAMILY_LABELS[fam_key]}  |  mAP@0.5",
                    color="white",
                    fontsize=11,
                    fontweight="bold",
                    pad=10,
                )

                ax.tick_params(colors="white")

                # Border color per family
                for _, sp in ax.spines.items():
                    sp.set_color(FAMILY_COLORS[fam_key])

                # -------------------------------------------------
                # Save Figure
                # -------------------------------------------------

                fname = out_dir / f"heatmap_{model_id}_{fam_key}.png"

                plt.savefig(
                    fname,
                    dpi=150,
                    bbox_inches="tight",
                    facecolor=fig.get_facecolor(),
                )

                plt.close()

                pbar.update(1)

    log.info(f"Heatmaps saved -> {out_dir}")


# ---------------------------------------------------------------------
# 2. Degradation Curves
# ---------------------------------------------------------------------

def plot_degradation_curves(
    df: pd.DataFrame,
    out_dir: str | Path = "results/degradation_curves",
) -> None:
    """
    Plot mAP@0.5 degradation curves across intensity levels.

    Each plot corresponds to one augmentation family.
    Each line corresponds to one model (M0-M5).

    X-axis:
        intensity level (0 -> 2)

    Y-axis:
        mean mAP@0.5
    """

    # -------------------------------------------------------------
    # Prepare Output Directory
    # -------------------------------------------------------------

    out_dir = Path(out_dir)
    ensure_dir(out_dir)

    models = sorted(df["model_id"].unique())

    # -------------------------------------------------------------
    # Generate One Plot Per Family
    # -------------------------------------------------------------

    for fam_key in tqdm(FAMILY_KEYS, desc="Degradation curves"):

        fig, ax = plt.subplots(figsize=(10, 6))

        fig.patch.set_facecolor("#0d0d1a")
        ax.set_facecolor("#111122")

        fdf = df[df["family"] == fam_key]

        # Plot each model's degradation curve
        for model_id in models:

            mdf = fdf[fdf["model_id"] == model_id]

            # Mean performance per intensity level
            means = mdf.groupby("row_idx")["map50"].mean().sort_index()

            ax.plot(
                means.index,
                means.values,
                marker="o",
                linewidth=2,
                color=MODEL_COLORS.get(model_id, "#ffffff"),
                label=model_id,
            )

        # Axis labels
        ax.set_xlabel(
            "Intensity Level (0=mild -> 2=strong)",
            color="white",
            fontsize=10,
        )

        ax.set_ylabel(
            "Mean mAP@0.5",
            color="white",
            fontsize=10,
        )

        # Title
        ax.set_title(
            f"Degradation Curve  -  {FAMILY_LABELS[fam_key]}",
            color="white",
            fontsize=12,
            fontweight="bold",
        )

        ax.tick_params(colors="white")

        ax.legend(
            facecolor="#1a1a2e",
            edgecolor=FAMILY_COLORS[fam_key],
            labelcolor="white",
            fontsize=9,
        )

        ax.grid(alpha=0.2, color="#444466")

        for sp in ax.spines.values():
            sp.set_color(FAMILY_COLORS[fam_key])

        # ---------------------------------------------------------
        # Save Plot
        # ---------------------------------------------------------

        fname = out_dir / f"degradation_{fam_key}.png"

        plt.savefig(
            fname,
            dpi=150,
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
        )

        plt.close()

    log.info(f"Degradation curves saved -> {out_dir}")


# ---------------------------------------------------------------------
# 3. Model Comparison Plot
# ---------------------------------------------------------------------

def plot_model_comparison(
    ranking_df: pd.DataFrame,
    out_path: str | Path = "results/model_comparison.png",
) -> None:
    """
    Create final comparison figure for all models.

    Left:
        Robustness score ranking

    Right:
        Clean vs Augmented mAP comparison
    """

    # -------------------------------------------------------------
    # Prepare Output Path
    # -------------------------------------------------------------

    out_path = Path(out_path)
    ensure_dir(out_path.parent)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    fig.patch.set_facecolor("#0d0d1a")

    for ax in axes:
        ax.set_facecolor("#111122")

    model_ids = ranking_df["model_id"].tolist()
    colors = [MODEL_COLORS.get(m, "#ffffff") for m in model_ids]

    # -------------------------------------------------------------
    # Left Plot: Robustness Score
    # -------------------------------------------------------------

    axes[0].barh(
        model_ids,
        ranking_df["robustness_score"],
        color=colors,
        edgecolor="#222233",
    )

    axes[0].set_xlabel(
        "Robustness Score",
        color="white",
    )

    axes[0].set_title(
        "Overall Robustness Score (higher = better)",
        color="white",
        fontweight="bold",
    )

    # -------------------------------------------------------------
    # Right Plot: Clean vs Augmented Performance
    # -------------------------------------------------------------

    x = np.arange(len(model_ids))
    w = 0.35

    axes[1].bar(
        x - w / 2,
        ranking_df["clean_map50"],
        width=w,
        label="Clean",
        color="#3498db",
    )

    axes[1].bar(
        x + w / 2,
        ranking_df["mean_map50_augmented"],
        width=w,
        label="Augmented",
        color="#e74c3c",
    )

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(model_ids)

    axes[1].set_ylabel(
        "mAP@0.5",
        color="white",
    )

    axes[1].set_title(
        "Clean vs Augmented Performance",
        color="white",
        fontweight="bold",
    )

    axes[1].legend(
        facecolor="#1a1a2e",
        edgecolor="#555577",
        labelcolor="white",
    )

    # -------------------------------------------------------------
    # Styling Cleanup
    # -------------------------------------------------------------

    for ax in axes:
        ax.tick_params(colors="white")
        ax.grid(alpha=0.15, color="#333355", axis="x")

        for sp in ax.spines.values():
            sp.set_color("#444466")

    plt.suptitle(
        "Phase-2 Model Robustness Comparison (M0-M5)",
        color="white",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )

    plt.tight_layout()

    # -------------------------------------------------------------
    # Save Final Figure
    # -------------------------------------------------------------

    plt.savefig(
        out_path,
        dpi=150,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )

    plt.close()

    log.info(f"Model comparison chart -> {out_path}")