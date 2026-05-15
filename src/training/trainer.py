"""
src/training/trainer.py
========================
Training pipeline for YOLOv8 augmentation experiments.

This module trains a single YOLOv8 model under a controlled augmentation
policy while ensuring experimental consistency across all model variants.

Experiment Design
-----------------
Two dataset strategies are used:

1. M0 (Baseline)
   - Uses the original KITTI dataset without augmentation.
   - Images are copied onto disk into a dedicated experiment directory.
   - Ultralytics loads data using its standard disk-based pipeline.
   - Serves as the clean reference model for comparison.

2. M1-M5 (Augmented)
   - Uses dynamic in-memory augmentation.
   - Original KITTI images remain untouched on disk.
   - Augmentations are applied per sample during loading.
   - Ultralytics Trainer.build_dataset() is monkey-patched so only the
     training split uses the augmented dataset.
   - Validation data remains unchanged for fair evaluation.

Pipeline Flow
--------------
1. Load training configuration and augmentation policy
2. Prepare dataset
      M0    -> original datasets/kitti/ (no copy)
      M1-M5 -> OnTheFlyDataset
3. Launch YOLOv8 training
4. Return TrainResult metadata

Research Goal
--------------
This architecture ensures:
- reproducibility
- fair augmentation comparison
- consistent validation protocol
- minimal disk overhead
- isolated augmentation policies
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

# torch imported lazily inside _train() to avoid startup errors

from src.augmentations.policies import get_policy, POLICIES
from src.training.dataset import (
    OnTheFlyDataset,
    build_yolo_dataset_class,
    write_data_yaml,
    write_m0_yaml,
)
from src.utils.io import load_yaml, ensure_dir
from src.utils.logger import get_logger

# ---------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------

log = get_logger(__name__)

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

# Baseline model identifier
BASELINE_MODEL = "M0"


# ---------------------------------------------------------------------
# Training Result Container
# ---------------------------------------------------------------------

@dataclass
class TrainResult:
    """
    Stores metadata produced after training completes.

    Attributes
    ----------
    model_id : str
        Experiment model identifier (M0-M5)

    best_pt : Path
        Path to best validation checkpoint

    run_dir : Path
        Ultralytics output directory

    elapsed_sec : float
        Total training runtime in seconds

    epochs : int
        Number of epochs completed
    """

    model_id: str
    best_pt: Path
    run_dir: Path
    elapsed_sec: float
    epochs: int


# ---------------------------------------------------------------------
# Main Trainer
# ---------------------------------------------------------------------

class ModelTrainer:
    """
    End-to-end trainer for a single YOLOv8 experiment model.

    Supported Models
    ----------------
    M0
        Baseline (no augmentation)

    M1-M5
        Augmented training variants using different augmentation policies

    Parameters
    ----------
    model_id : str
        Experiment identifier ("M0" - "M5")

    cfg_path : str | Path
        Path to training configuration YAML

    overrides : dict | None
        Optional hyperparameter overrides
    """

    def __init__(
        self,
        model_id: str,
        cfg_path: str | Path = "configs/train.yaml",
        overrides: dict | None = None,
    ):
        """
        Initialize trainer configuration and augmentation policy.
        """

        # -------------------------------------------------------------
        # Store experiment identity
        # -------------------------------------------------------------

        self.model_id = model_id

        # -------------------------------------------------------------
        # Load YAML configuration
        # -------------------------------------------------------------

        self.cfg = load_yaml(cfg_path)

        # Runtime overrides allow quick experiment adjustments
        # without modifying the original YAML file.
        self.overrides = overrides or {}

        # Merge default hyperparameters with overrides
        self.hp = {
            **self.cfg["hyperparams"],
            **self.overrides,
        }

        # Dataset-related configuration
        self.data_cfg = self.cfg["data"]

        # Output/logging configuration
        self.out_cfg = self.cfg["output"]

        # -------------------------------------------------------------
        # Validate experiment identifier
        # -------------------------------------------------------------

        if model_id not in self.cfg["models"]:
            raise ValueError(
                f"model_id '{model_id}' not found in configs/train.yaml. "
                f"Valid models: {list(self.cfg['models'])}"
            )

        # -------------------------------------------------------------
        # Load augmentation policy
        # -------------------------------------------------------------

        self.policy_fn = get_policy(model_id)

        # -------------------------------------------------------------
        # Log initialization summary
        # -------------------------------------------------------------

        log.info(
            f"ModelTrainer ready - {model_id} "
            f"({POLICIES[model_id]['name']}, "
            f"pool={POLICIES[model_id]['pool_size']} intensities)"
        )

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def run(self, overwrite_aug: bool = False) -> TrainResult:
        """
        Execute the full training pipeline.

        Parameters
        ----------
        overwrite_aug : bool
            If True, regenerates baseline disk dataset even if it exists.

        Returns
        -------
        TrainResult
            Training metadata and checkpoint paths.
        """

        # Track total runtime
        t0 = time.time()

        # -------------------------------------------------------------
        # Dataset Preparation
        # -------------------------------------------------------------

        # M0 uses a disk-based dataset copy.
        # M1-M5 use dynamic in-memory augmentation.
        if self.model_id == BASELINE_MODEL:

            log.info(
                f"[{self.model_id}] Step 1/3 - "
                f"M0 baseline: using original datasets/kitti/ directly"
            )

            data_yaml = self._prepare_m0_dataset()
            otf = None

        else:
            log.info(
                f"[{self.model_id}] Step 1/3 - "
                f"Build in-memory dataset (no disk writes)"
            )

            data_yaml, otf = self._prepare_otf_dataset()

        # -------------------------------------------------------------
        # YOLOv8 Training
        # -------------------------------------------------------------

        log.info(f"[{self.model_id}] Step 2/3 - Train YOLOv8")

        run_dir = self._train(data_yaml, otf)

        # -------------------------------------------------------------
        # Finalize Results
        # -------------------------------------------------------------

        best_pt = run_dir / "weights" / "best.pt"

        elapsed = time.time() - t0

        log.info(
            f"[{self.model_id}] Step 3/3 - Done "
            f"best.pt={best_pt} "
            f"({elapsed / 60:.1f} min)"
        )

        return TrainResult(
            model_id=self.model_id,
            best_pt=best_pt,
            run_dir=run_dir,
            elapsed_sec=elapsed,
            epochs=self.hp["epochs"],
        )

    # -----------------------------------------------------------------
    # Dataset Preparation
    # -----------------------------------------------------------------

    def _prepare_m0_dataset(self) -> Path:
        """
        Prepare M0 baseline dataset.

        Points directly at datasets/kitti/kitti.yaml with no copying.
        No kitti_aug/ folder is created for M0.

        Returns
        -------
        Path
            Path to the original kitti.yaml.
        """
        return write_m0_yaml(
            kitti_yaml_path=Path(self.data_cfg["data_yaml"])
        )

    def _prepare_otf_dataset(self):
        """
        Prepare in-memory augmented dataset (M1-M5).

        No augmented images are written to disk.
        Augmentation occurs dynamically during sample loading.

        Returns
        -------
        tuple
            (data_yaml_path, OnTheFlyDataset instance)
        """

        # Original KITTI root
        kitti_root = Path(self.data_cfg["kitti_root"])

        # -------------------------------------------------------------
        # Build dynamic augmentation dataset
        # -------------------------------------------------------------

        otf = OnTheFlyDataset(
            kitti_root=kitti_root,
            split="train",

            # Augmentation policy specific to this model
            policy=self.policy_fn,

            model_id=self.model_id,

            # Fixed seed improves reproducibility
            base_seed=self.hp["seed"],

            img_size=self.data_cfg["img_size"],
        )

        log.info(
            f"[{self.model_id}] OnTheFlyDataset ready - "
            f"{len(otf)} images, "
            f"policy={POLICIES[self.model_id]['name']}"
        )

        # -------------------------------------------------------------
        # Generate temporary YOLO data YAML
        # -------------------------------------------------------------

        data_yaml = write_data_yaml(
            kitti_yaml_path=Path(self.data_cfg["data_yaml"]),
            model_id=self.model_id,
        )

        log.info(f"[{self.model_id}] Data yaml -> {data_yaml}")

        return data_yaml, otf

    # -----------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------

    def _train(
        self,
        data_yaml: Path,
        otf_dataset: OnTheFlyDataset | None,
    ) -> Path:
        """
        Launch YOLOv8 training.

        If an OnTheFlyDataset is provided, Ultralytics'
        Trainer.build_dataset() is monkey-patched so the training split
        uses dynamic augmentation while validation remains untouched.

        Parameters
        ----------
        data_yaml : Path
            Path to YOLO dataset YAML

        otf_dataset : OnTheFlyDataset | None
            Dynamic augmentation dataset wrapper

        Returns
        -------
        Path
            YOLO output directory
        """

        # -------------------------------------------------------------
        # Lazy imports
        # -------------------------------------------------------------

        # Delaying import improves startup performance and prevents
        # unnecessary dependency loading when not training.
        try:
            from ultralytics import YOLO

        except ImportError:
            raise ImportError(
                "ultralytics is not installed. "
                "Run: pip install ultralytics"
            )

        # -------------------------------------------------------------
        # Initialize YOLOv8 model
        # -------------------------------------------------------------

        # Example:
        # yolov8n.pt
        # yolov8s.pt
        # yolov8m.pt
        model = YOLO(self.cfg["yolo_variant"])

        # -------------------------------------------------------------
        # Monkey-Patch Dataset Builder (M1-M5)
        # -------------------------------------------------------------

        # Only augmentation models require dataset interception.
        # Baseline model uses Ultralytics defaults unchanged.
        if otf_dataset is not None:

            # Build YOLO-compatible dataset wrapper class
            AugDataset = build_yolo_dataset_class(otf_dataset)

            # Store original implementation for validation split fallback
            _orig_build = None

            def _patched_build_dataset(
                trainer_self,
                img_path,
                mode="train",
                batch=None,
            ):
                """
                Intercepts Ultralytics dataset creation.

                Training split:
                    Uses augmented in-memory dataset

                Validation/Test:
                    Falls back to original Ultralytics behavior
                """

                # -----------------------------------------------------
                # Replace TRAIN dataset only
                # -----------------------------------------------------

                if mode == "train":

                    log.info(
                        f"[{self.model_id}] "
                        f"build_dataset(train) -> "
                        f"OnTheFlyDataset "
                        f"({len(otf_dataset)} images, "
                        f"policy={POLICIES[self.model_id]['name']})"
                    )

                    return AugDataset(
                        img_path,

                        # Target image size
                        imgsz=trainer_self.args.imgsz,

                        # Batch size from trainer args
                        batch_size=trainer_self.args.batch,

                        # Disable native YOLO augmentation because
                        # augmentation is already handled externally
                        augment=False,

                        hyp=trainer_self.args,

                        # Rectangular batching disabled to preserve
                        # augmentation consistency
                        rect=False,

                        # Optional caching
                        cache=trainer_self.args.cache or False,

                        single_cls=trainer_self.args.single_cls or False,

                        # Model stride alignment
                        stride=int(trainer_self.stride),

                        # No additional padding
                        pad=0.0,

                        prefix="train: ",

                        task=trainer_self.args.task,

                        classes=trainer_self.args.classes,

                        data=trainer_self.data,

                        fraction=trainer_self.args.fraction,
                    )

                # -----------------------------------------------------
                # Validation/Test use original implementation
                # -----------------------------------------------------

                return _orig_build(
                    trainer_self,
                    img_path,
                    mode=mode,
                    batch=batch,
                )

            def _on_pretrain_routines_end(trainer):
                """
                Callback executed before training starts.

                At this stage Ultralytics Trainer exists, so we can safely
                replace its dataset builder method.
                """

                nonlocal _orig_build

                # Save original implementation
                _orig_build = trainer.__class__.build_dataset

                # Inject patched implementation
                trainer.__class__.build_dataset = (
                    _patched_build_dataset
                )

            # Register callback into YOLO lifecycle
            model.add_callback(
                "on_pretrain_routines_end",
                _on_pretrain_routines_end,
            )

        # -------------------------------------------------------------
        # YOLO Augmentation Overrides
        # -------------------------------------------------------------

        # Additional Ultralytics augmentation settings
        # may be disabled or adjusted globally.
        aug_overrides = self.cfg.get("yolo_aug_overrides", {})

        # -------------------------------------------------------------
        # Training Configuration
        # -------------------------------------------------------------

        train_kwargs = dict(

            # Dataset configuration
            data=str(data_yaml),

            # Input image resolution
            imgsz=self.data_cfg["img_size"],

            # Training schedule
            epochs=self.hp["epochs"],
            batch=self.hp["batch_size"],

            # ---------------------------------------------------------
            # Optimizer Settings
            # ---------------------------------------------------------

            lr0=self.hp["lr0"],
            lrf=self.hp["lrf"],
            momentum=self.hp["momentum"],
            weight_decay=self.hp["weight_decay"],

            # ---------------------------------------------------------
            # Warmup
            # ---------------------------------------------------------

            warmup_epochs=self.hp["warmup_epochs"],
            warmup_momentum=self.hp["warmup_momentum"],
            warmup_bias_lr=self.hp["warmup_bias_lr"],

            optimizer=self.hp["optimizer"],

            # ---------------------------------------------------------
            # Reproducibility
            # ---------------------------------------------------------

            # Shared seed across all experiments ensures
            # fair augmentation comparisons.
            seed=self.hp["seed"],

            # ---------------------------------------------------------
            # System Configuration
            # ---------------------------------------------------------

            workers=self.hp["workers"],
            device=self.hp["device"],

            # ---------------------------------------------------------
            # Output Directories
            # ---------------------------------------------------------

            project=str(
                Path(self.out_cfg["project"]).resolve()
            ),

            name=self.model_id,

            exist_ok=self.out_cfg["exist_ok"],

            verbose=self.out_cfg["verbose"],

            # ---------------------------------------------------------
            # Checkpoint Saving
            # ---------------------------------------------------------

            save_period=self.hp["save_period"],

            # ---------------------------------------------------------
            # External YOLO Overrides
            # ---------------------------------------------------------

            **aug_overrides,
        )

        # -------------------------------------------------------------
        # Log full training configuration
        # -------------------------------------------------------------

        log.info(
            f"[{self.model_id}] train kwargs: {train_kwargs}"
        )

        # -------------------------------------------------------------
        # Launch Training
        # -------------------------------------------------------------

        results = model.train(**train_kwargs)

        # -------------------------------------------------------------
        # Resolve Actual Output Directory
        # -------------------------------------------------------------

        # Ultralytics may auto-increment experiment folders,
        # so we use the runtime-generated path directly.
        run_dir = Path(results.save_dir)

        log.info(
            f"[{self.model_id}] YOLO output dir -> {run_dir}"
        )

        return run_dir