"""
src/training/config.py
=======================
Centralized training configuration for Phase-2 YOLOv8 experiments.

This module defines a frozen configuration object shared across all
experiment models (M0-M5).

Research Constraints
--------------------
According to the Phase-2 experimental protocol:

- All models must use the SAME YOLO architecture
- All models must use the SAME dataset split
- All models must use the SAME hyperparameters
- The ONLY changing variable is augmentation policy

This ensures:
- fair comparison
- reproducibility
- controlled experimentation
- scientifically valid augmentation analysis

Configuration Philosophy
------------------------
The TrainConfig dataclass is immutable (frozen=True) to prevent
accidental runtime modification of experiment parameters.

This is important for:
- reproducibility
- experiment tracking
- debugging
- research integrity
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------
# Frozen Training Configuration
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class TrainConfig:
    """
    Immutable training configuration shared across all models.

    Parameters
    ----------
    model_id : str
        Experiment identifier ("M0" - "M5")

    Notes
    -----
    All hyperparameters remain identical across experiments.
    Only augmentation policy changes between models.
    """

    # -----------------------------------------------------------------
    # Model Configuration
    # -----------------------------------------------------------------

    # Experiment identifier
    model_id: str

    # YOLOv8 model variant
    #
    # yolov8n.pt:
    #   Nano version (lightweight and fast)
    #
    # Alternative options:
    #   yolov8s.pt
    #   yolov8m.pt
    #   yolov8l.pt
    #   yolov8x.pt
    #
    # Nano is preferred for controlled augmentation studies
    # because it reduces training cost and iteration time.
    yolo_variant: str = "yolov8n.pt"

    # -----------------------------------------------------------------
    # Dataset Configuration
    # -----------------------------------------------------------------

    # Path to KITTI dataset YAML
    data_yaml: str = "datasets/kitti/kitti.yaml"

    # Input image resolution
    #
    # 640 is the standard YOLOv8 default and provides
    # a strong balance between:
    # - accuracy
    # - GPU memory usage
    # - training speed
    img_size: int = 640

    # -----------------------------------------------------------------
    # Core Training Hyperparameters
    # -----------------------------------------------------------------

    # Number of full training passes over dataset
    epochs: int = 50

    # Samples processed simultaneously per iteration
    batch_size: int = 16

    # -------------------------------------------------------------
    # Optimizer Parameters
    # -------------------------------------------------------------

    # Initial learning rate
    lr0: float = 0.01

    # Final learning rate scaling factor
    #
    # Final LR = lr0 * lrf
    lrf: float = 0.01

    # SGD momentum
    #
    # Helps accelerate convergence and stabilize updates.
    momentum: float = 0.937

    # L2 regularization
    #
    # Reduces overfitting by penalizing large weights.
    weight_decay: float = 0.0005

    # -----------------------------------------------------------------
    # Warmup Parameters
    # -----------------------------------------------------------------

    # Number of warmup epochs
    #
    # Warmup stabilizes early training by gradually increasing
    # learning rate from small values.
    warmup_epochs: float = 3.0

    # Initial momentum during warmup phase
    warmup_momentum: float = 0.8

    # Initial bias learning rate during warmup
    warmup_bias_lr: float = 0.1

    # -----------------------------------------------------------------
    # Optimizer Selection
    # -----------------------------------------------------------------

    # Stochastic Gradient Descent is commonly used in YOLO training
    # and offers stable convergence characteristics.
    optimizer: str = "SGD"

    # -----------------------------------------------------------------
    # Reproducibility
    # -----------------------------------------------------------------

    # Global training seed
    #
    # Shared seed ensures:
    # - deterministic behavior
    # - reproducibility
    # - fair augmentation comparison
    seed: int = 42

    # -----------------------------------------------------------------
    # Ultralytics Built-In Augmentations
    # -----------------------------------------------------------------

    # IMPORTANT:
    # All native YOLO augmentations are disabled.
    #
    # Reason:
    # This project uses externally controlled augmentation policies.
    #
    # Enabling YOLO augmentations would introduce uncontrolled
    # experimental variables and invalidate comparisons.

    # Mosaic augmentation disabled
    mosaic: float = 0.0

    # MixUp augmentation disabled
    mixup: float = 0.0

    # Copy-paste augmentation disabled
    copy_paste: float = 0.0

    # -------------------------------------------------------------
    # Geometric Transformations
    # -------------------------------------------------------------

    # Rotation disabled
    #
    # Rotation is controlled manually via augmentation policy.
    degrees: float = 0.0

    # Translation disabled
    translate: float = 0.0

    # Scale augmentation disabled
    scale: float = 0.0

    # Shear transformation disabled
    shear: float = 0.0

    # Perspective warp disabled
    perspective: float = 0.0

    # -------------------------------------------------------------
    # Flip Augmentations
    # -------------------------------------------------------------

    # Vertical flipping disabled
    flipud: float = 0.0

    # Horizontal flipping disabled
    fliplr: float = 0.0

    # -------------------------------------------------------------
    # HSV Color Augmentations
    # -------------------------------------------------------------

    # Hue augmentation disabled
    hsv_h: float = 0.0

    # Saturation augmentation disabled
    hsv_s: float = 0.0

    # Brightness augmentation disabled
    hsv_v: float = 0.0

    # -----------------------------------------------------------------
    # Output & System Configuration
    # -----------------------------------------------------------------

    # Experiment output directory
    project: str = "experiments"

    # Number of PyTorch dataloader workers
    workers: int = 4

    # Training device
    #
    # Examples:
    #   "0"    -> GPU 0
    #   "0,1"  -> Multi-GPU
    #   "cpu"  -> CPU-only training
    device: str = "0"

    # Allow overwriting existing experiment folders
    exist_ok: bool = True

    # Enable verbose logging
    verbose: bool = True

    # Save checkpoint every N epochs
    save_period: int = 10

    # -----------------------------------------------------------------
    # Derived Properties
    # -----------------------------------------------------------------

    @property
    def name(self) -> str:
        """
        Generate unique Ultralytics run name.

        Returns
        -------
        str
            Experiment-specific run identifier

        Example
        -------
        M1_controlled
        """

        return f"{self.model_id}_controlled"

    # -----------------------------------------------------------------
    # Ultralytics Export Helper
    # -----------------------------------------------------------------

    def to_ultralytics_kwargs(self) -> dict:
        """
        Convert configuration into Ultralytics-compatible kwargs.

        Returns
        -------
        dict
            Keyword arguments for:
                model.train(**kwargs)

        Notes
        -----
        Fields not recognized by Ultralytics are removed:
        - model_id
        - yolo_variant

        The generated experiment name is injected automatically.
        """

        # -------------------------------------------------------------
        # Convert Dataclass -> Dictionary
        # -------------------------------------------------------------

        d = asdict(self)

        # -------------------------------------------------------------
        # Remove Internal-Only Fields
        # -------------------------------------------------------------

        # model_id is used only for experiment tracking
        d.pop("model_id")

        # Model weights are loaded separately during initialization
        d.pop("yolo_variant")

        # -------------------------------------------------------------
        # Inject Generated Experiment Name
        # -------------------------------------------------------------

        d["name"] = self.name

        return d


# ---------------------------------------------------------------------
# Factory Function
# ---------------------------------------------------------------------

def get_train_config(
    model_id: str,
    **overrides,
) -> TrainConfig:
    """
    Create a TrainConfig instance with optional overrides.

    Parameters
    ----------
    model_id : str
        Experiment identifier

    **overrides
        Optional field overrides

    Returns
    -------
    TrainConfig
        Immutable training configuration

    Examples
    --------
    Default configuration:

    >>> cfg = get_train_config("M1")

    Override epochs:

    >>> cfg = get_train_config("M2", epochs=30)

    CPU-only training:

    >>> cfg = get_train_config(
    ...     "M3",
    ...     device="cpu"
    ... )
    """

    # -------------------------------------------------------------
    # Extract Default Dataclass Values
    # -------------------------------------------------------------

    # Build dictionary of default field values excluding model_id.
    base_fields = {
        f.name: f.default
        for f in TrainConfig.__dataclass_fields__.values()
        if f.name != "model_id"
    }

    # -------------------------------------------------------------
    # Apply Runtime Overrides
    # -------------------------------------------------------------

    # Overrides allow quick experimentation without modifying
    # source configuration files.
    base_fields.update(overrides)

    # -------------------------------------------------------------
    # Construct Immutable Config
    # -------------------------------------------------------------

    return TrainConfig(
        model_id=model_id,
        **base_fields,
    )