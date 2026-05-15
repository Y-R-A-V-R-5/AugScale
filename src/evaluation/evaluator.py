"""
src/evaluation/evaluator.py
============================
Robustness evaluation engine for Phase-2 YOLOv8 experiments.

Fixes applied
-------------
1. Labels copied alongside images in temp dir so YOLO finds them.
2. YAML val: points to tmpdir root (not images/) -- Ultralytics expects
   the parent folder containing images/ and labels/ subdirectories.
3. model.val() output redirected to experiments/<model_id>/eval/
   instead of the default runs/detect/ folder.
4. save=False + plots=False suppress all file writes from YOLO val.
5. Both CSVs (per-model + global) are updated after EVERY condition,
   not just at the end -- crash-safe incremental writing.
6. YOLO runs/ folder creation suppressed via absolute project path
   and name="." so outputs land directly in experiments/<model_id>/eval/.
"""

from __future__ import annotations

import shutil
import time
import tempfile
import cv2
import numpy as np
import pandas as pd

from pathlib import Path
from typing import List, Optional

from tqdm import tqdm

from src.augmentations.transforms import apply_intensity
from src.evaluation.conditions import EvalCondition, build_eval_conditions
from src.evaluation.metrics import MetricsCollector
from src.utils.io import load_yaml, ensure_dir
from src.utils.logger import get_logger


log = get_logger(__name__)

IMG_EXTS = ("*.png", "*.jpg", "*.jpeg")


class ModelEvaluator:
    """
    Executes full robustness evaluation for one trained model (M0-M5).

    Evaluation conditions: 37 total
        1  clean baseline (no augmentation)
       36  corruption conditions (4 families x 3 sub-augs x 3 seeds)

    Both CSVs are written after every single condition so progress is
    never lost if the run is interrupted mid-way.

    Parameters
    ----------
    model_id    : "M0" | "M1" | "M2" | "M3" | "M4" | "M5"
    weights_pt  : path to trained best.pt
    cfg_path    : configs/evaluation.yaml
    train_cfg   : configs/train.yaml  (for experiments/ output path)
    """

    def __init__(
        self,
        model_id: str,
        weights_pt: str | Path,
        cfg_path: str | Path = "configs/evaluation.yaml",
        train_cfg: str | Path = "configs/train.yaml",
    ):
        self.model_id   = model_id
        self.weights_pt = Path(weights_pt)
        self.cfg        = load_yaml(cfg_path)
        self.train_cfg  = load_yaml(train_cfg)
        self.eval_data  = self.cfg["eval_data"]
        self.out_cfg    = self.cfg["output"]

        if not self.weights_pt.exists():
            raise FileNotFoundError(
                f"Weights not found: {self.weights_pt}\n"
                f"Train first: python scripts/train.py --model {model_id}"
            )

        self.conditions: List[EvalCondition] = build_eval_conditions(cfg_path)
        self.val_images  = self._collect_val_images()
        self.val_labels  = self._collect_val_labels()

        # ----------------------------------------------------------------
        # eval output dir: <PROJECT_ROOT>/experiments/<model_id>/eval/
        #
        # MUST be absolute. Ultralytics resolves a relative `project`
        # arg against its own internal CWD, producing the double-path
        # bug:  runs/detect/experiments/.../eval/
        # Passing an absolute path bypasses that logic entirely.
        # ----------------------------------------------------------------
        _exp_rel = Path(self.train_cfg["output"]["project"])  # e.g. "./experiments"
        _repo    = Path(__file__).resolve().parent.parent.parent  # repo root
        self._eval_out_dir = (_repo / _exp_rel / model_id / "eval").resolve()
        ensure_dir(self._eval_out_dir)

        # Resolved CSV paths — set once, used throughout the run
        self._per_model_csv = self._eval_out_dir / "raw_eval.csv"
        self._global_csv    = Path(self.out_cfg["raw_csv"])
        ensure_dir(self._global_csv.parent)

        log.info(
            f"ModelEvaluator ready | {model_id} | "
            f"{len(self.val_images)} val images | "
            f"{len(self.conditions)} conditions | "
            f"eval_out={self._eval_out_dir}"
        )

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def run(self) -> pd.DataFrame:
        """
        Run all 37 evaluation conditions.

        Both CSVs are updated after every condition:
            experiments/<model_id>/eval/raw_eval.csv  — per-model, overwritten fresh each run
            results/raw_eval.csv                       — global, rows appended incrementally

        Returns the full DataFrame for this model once all conditions finish.
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("Run: pip install ultralytics")

        model     = YOLO(str(self.weights_pt))
        collector = MetricsCollector(self.model_id)
        t0        = time.time()

        # Clean baseline map50 — set after condition 0 so delta can be
        # computed immediately for every subsequent condition.
        baseline_map50: Optional[float] = None

        # ------------------------------------------------------------------
        # Wipe this model's per-run CSV so we start fresh (no stale rows
        # from a previous partial run).  The global CSV is append-only so
        # older models are never touched.
        # ------------------------------------------------------------------
        if self._per_model_csv.exists():
            self._per_model_csv.unlink()

        log.info(f"[{self.model_id}] Starting robustness evaluation ...")

        for cond in tqdm(
            self.conditions,
            desc=f"Eval {self.model_id}",
            unit="cond",
        ):
            # ---------------------------------------------------------
            # Run YOLO val for this single condition
            # ---------------------------------------------------------
            metrics = self._eval_condition(model, cond)
            collector.add(cond, metrics)

            # ---------------------------------------------------------
            # Compute delta_map50 immediately
            #   condition 0  → baseline reference, delta = 0.0
            #   condition 1+ → delta vs the stored baseline
            # ---------------------------------------------------------
            if cond.condition_id == 0:
                baseline_map50 = metrics["map50"]

            delta = (
                metrics["map50"] - baseline_map50
                if baseline_map50 is not None
                else float("nan")
            )

            # ---------------------------------------------------------
            # Build a single-row DataFrame for this condition
            # ---------------------------------------------------------
            row_df = collector.latest_row_as_dataframe(delta)

            # ---------------------------------------------------------
            # Write to both CSVs immediately after this condition
            # ---------------------------------------------------------
            self._append_condition(row_df)

            log.debug(
                f"[{self.model_id}] cond {cond.condition_id:02d} "
                f"map50={metrics['map50']:.4f}  delta={delta:+.4f}"
            )

        # ------------------------------------------------------------------
        # Build and return the full per-model DataFrame from the collector
        # (delta_map50 is already correct on every row at this point)
        # ------------------------------------------------------------------
        df = collector.to_dataframe()

        # Re-apply delta from stored baseline to ensure the in-memory
        # DataFrame is consistent with what was written to disk.
        if baseline_map50 is not None:
            df["delta_map50"] = df["map50"] - baseline_map50

        elapsed = time.time() - t0
        log.info(
            f"[{self.model_id}] Done | "
            f"{len(self.conditions)} conditions | {elapsed / 60:.1f} min"
        )

        return df

    # -------------------------------------------------------------------------
    # Dataset helpers
    # -------------------------------------------------------------------------

    def _collect_val_images(self) -> List[Path]:
        val_dir = (
            Path(self.eval_data["kitti_root"])
            / "images"
            / self.eval_data["split"]
        )
        imgs: List[Path] = []
        for ext in IMG_EXTS:
            imgs.extend(sorted(val_dir.glob(ext)))
        if not imgs:
            raise FileNotFoundError(f"No val images found in {val_dir}")
        return imgs

    def _collect_val_labels(self) -> List[Path]:
        """
        Return label paths parallel to val_images.
        Labels live in datasets/kitti/labels/val/<stem>.txt
        """
        label_dir = (
            Path(self.eval_data["kitti_root"])
            / "labels"
            / self.eval_data["split"]
        )
        labels: List[Path] = []
        for img_path in self.val_images:
            lbl = label_dir / (img_path.stem + ".txt")
            labels.append(lbl)   # may not exist for background images
        return labels

    # -------------------------------------------------------------------------
    # Per-condition evaluation
    # -------------------------------------------------------------------------

    def _eval_condition(self, model, cond: EvalCondition) -> dict:
        """
        Evaluate one condition inside a self-contained temp directory.

        YOLO output is directed to experiments/<model_id>/eval/ using an
        absolute project path and name="." so it never touches runs/detect/.

        Temp layout:
            tmpdir/
              images/   <-- (optionally corrupted) val images
              labels/   <-- copied from datasets/kitti/labels/val/
              data.yaml <-- val: <tmpdir>
        """
        import yaml as _yaml

        with tempfile.TemporaryDirectory(prefix="phase2_eval_") as _tmpdir:
            tmpdir      = Path(_tmpdir)
            tmp_img_dir = tmpdir / "images"
            tmp_lbl_dir = tmpdir / "labels"
            tmp_img_dir.mkdir(parents=True, exist_ok=True)
            tmp_lbl_dir.mkdir(parents=True, exist_ok=True)

            # ------------------------------------------------------------------
            # Write (optionally corrupted) images + copy labels
            # ------------------------------------------------------------------
            for img_path, lbl_path in zip(self.val_images, self.val_labels):
                bgr = cv2.imread(str(img_path))
                if bgr is None:
                    continue

                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

                if cond.intensity is not None:
                    rgb = apply_intensity(rgb, cond.intensity, seed=cond.seed)

                cv2.imwrite(
                    str(tmp_img_dir / img_path.name),
                    cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                )

                if lbl_path.exists():
                    shutil.copy2(lbl_path, tmp_lbl_dir / lbl_path.name)

            # ------------------------------------------------------------------
            # Build YOLO data yaml
            # ------------------------------------------------------------------
            kitti_yaml = Path(self.eval_data["kitti_root"]) / (
                Path(self.eval_data["kitti_root"]).name + ".yaml"
            )
            if not kitti_yaml.exists():
                kitti_yaml = Path("datasets/kitti/kitti.yaml")

            with open(kitti_yaml, "r", encoding="utf-8", errors="replace") as f:
                data_cfg = _yaml.safe_load(f)

            data_cfg["val"] = str(tmpdir)

            tmp_yaml = tmpdir / "data.yaml"
            with open(tmp_yaml, "w", encoding="utf-8") as f:
                _yaml.dump(data_cfg, f, default_flow_style=False, allow_unicode=True)

            # ------------------------------------------------------------------
            # YOLO val — direct output into experiments/<model_id>/eval/
            #
            # Two rules stop runs/detect/ from being created:
            #
            #   1. project must be ABSOLUTE.  A relative path is appended
            #      to YOLO's internal default base (runs/detect/), producing
            #      the double-path:  runs/detect/experiments/.../eval/
            #      An absolute path is used as-is.
            #
            #   2. name="."  YOLO computes save_dir = project / name, so
            #      "." keeps save_dir == project:
            #          experiments/<model_id>/eval/        ← correct
            #      instead of:
            #          experiments/<model_id>/eval/cond_00/ ← extra subdir
            #
            # save=False + plots=False: suppress image/plot file writes.
            # exist_ok=True: don't error if the dir already exists.
            # ------------------------------------------------------------------
            results = model.val(
                data     = str(tmp_yaml),
                imgsz    = self.eval_data["img_size"],
                batch    = self.eval_data["batch_size"],
                device   = self.eval_data["device"],
                verbose  = False,
                save     = False,
                plots    = False,
                project  = str(self._eval_out_dir),  # absolute — no runs/detect/ prefix
                name     = ".",                       # flat: save_dir == project
                exist_ok = True,
            )

        return {
            "map50":     float(results.box.map50),
            "map50_95":  float(results.box.map),
            "precision": float(results.box.mp),
            "recall":    float(results.box.mr),
        }

    # -------------------------------------------------------------------------
    # Incremental CSV writing
    # -------------------------------------------------------------------------

    def _append_condition(self, row_df: pd.DataFrame) -> None:
        """
        Append a single condition row to both CSVs immediately after it
        completes.  Called 37 times per model — once per condition.

        Per-model CSV  (experiments/<model_id>/eval/raw_eval.csv)
            Wiped at the start of each run, then rows appended one by one.
            Always reflects exactly the conditions completed so far.

        Global CSV  (results/raw_eval.csv)
            Shared across all models — header written only when the file
            does not yet exist, rows appended.  Rows from previous models
            are never modified.
        """
        # ------------------------------------------------------------------
        # 1. Per-model CSV — append (file was wiped at run start)
        # ------------------------------------------------------------------
        header_per = not self._per_model_csv.exists()
        row_df.to_csv(
            self._per_model_csv,
            mode    = "a",
            header  = header_per,
            index   = False,
            encoding= "utf-8",
        )

        # ------------------------------------------------------------------
        # 2. Global aggregated CSV — append
        # ------------------------------------------------------------------
        header_global = not self._global_csv.exists()
        row_df.to_csv(
            self._global_csv,
            mode    = "a",
            header  = header_global,
            index   = False,
            encoding= "utf-8",
        )
