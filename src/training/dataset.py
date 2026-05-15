"""
src/training/dataset.py
========================
Dataset utilities for Phase-2 YOLOv8 augmentation experiments.

Dataset Strategies
------------------
M0  (Baseline)
    Points DIRECTLY at datasets/kitti/ with no copying or intermediate
    directories. No kitti_aug/ folder is created for M0.

M1-M5  (Augmented)
    OnTheFlyDataset loads original KITTI images and applies the model's
    augmentation policy dynamically in memory. No augmented images are
    written to disk.

Ultralytics Integration
-----------------------
build_yolo_dataset_class() creates a YOLODataset subclass that
intercepts load_image() for the training split only. Validation
always reads directly from datasets/kitti/.

Data YAML Generation
---------------------
write_data_yaml() writes a minimal YOLO-compatible YAML that points
train + val at the original KITTI paths. For M1-M5 the train image
pixels are intercepted in memory; for M0 they are read straight off
disk with no detour.
"""

from __future__ import annotations

import cv2
from pathlib import Path
from typing import Callable, List, Optional

import yaml
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Supported extensions
# ---------------------------------------------------------------------------

IMG_EXTS = ("*.png", "*.jpg", "*.jpeg", "*.bmp")


def _collect_images(split_dir: Path) -> List[Path]:
    """Return a sorted list of all images in split_dir."""
    files: List[Path] = []
    for ext in IMG_EXTS:
        files.extend(sorted(split_dir.glob(ext)))
    return files


# ---------------------------------------------------------------------------
# M0 -- direct KITTI YAML (no copying, no kitti_aug/)
# ---------------------------------------------------------------------------

def write_m0_yaml(kitti_yaml_path: Path) -> Path:
    """
    Return the original kitti.yaml unchanged.

    M0 trains directly from datasets/kitti/ -- no files are copied,
    no kitti_aug/ folder is created.

    Parameters
    ----------
    kitti_yaml_path : Path
        Path to datasets/kitti/kitti.yaml

    Returns
    -------
    Path
        The same path, passed straight through to YOLO.
    """
    kitti_yaml_path = Path(kitti_yaml_path)

    if not kitti_yaml_path.exists():
        raise FileNotFoundError(
            f"KITTI YAML not found: {kitti_yaml_path}\n"
            "Expected at: datasets/kitti/kitti.yaml"
        )

    print(f"  [M0] Using original dataset directly -> {kitti_yaml_path}")
    return kitti_yaml_path


# ---------------------------------------------------------------------------
# M1-M5 -- On-The-Fly augmented dataset (in-memory, no disk writes)
# ---------------------------------------------------------------------------

class OnTheFlyDataset:
    """
    Loads original KITTI images and applies an augmentation policy
    dynamically during training. No augmented files are written to disk.

    Parameters
    ----------
    kitti_root : Path
        Root KITTI directory (datasets/kitti/)
    split : str
        "train" or "val"
    policy : Callable
        (rgb_uint8, seed: int) -> augmented_rgb_uint8
    model_id : str
        Experiment ID (M1-M5)
    base_seed : int
        Seed offset; sample i uses seed = base_seed + i
    img_size : int
        Training image resolution (used for reference only)
    """

    def __init__(
        self,
        kitti_root: Path,
        split: str,
        policy: Callable,
        model_id: str,
        base_seed: int = 42,
        img_size: int = 640,
    ):
        self.kitti_root = Path(kitti_root)
        self.split      = split
        self.policy     = policy
        self.model_id   = model_id
        self.base_seed  = base_seed
        self.img_size   = img_size

        img_dir      = self.kitti_root / "images" / split
        self.images  = _collect_images(img_dir)

        if not self.images:
            raise FileNotFoundError(
                f"No images found in {img_dir}. "
                "Check datasets/kitti/images/train/ exists."
            )

        self.label_dir = self.kitti_root / "labels" / split

        # O(1) filename -> index lookup for Ultralytics hook
        self.name_to_idx: dict = {
            p.name: i for i, p in enumerate(self.images)
        }

    def __len__(self) -> int:
        return len(self.images)

    def load_image(self, index: int):
        """
        Load image at index, apply augmentation, return BGR ndarray.

        Returns
        -------
        tuple  (augmented_bgr, orig_height, orig_width)
        """
        img_path = self.images[index]

        bgr = cv2.imread(str(img_path))
        if bgr is None:
            raise IOError(f"Cannot read image: {img_path}")

        orig_h, orig_w = bgr.shape[:2]

        seed    = self.base_seed + index
        rgb     = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        aug_rgb = self.policy(rgb, seed)
        aug_bgr = cv2.cvtColor(aug_rgb, cv2.COLOR_RGB2BGR)

        return aug_bgr, orig_h, orig_w


# ---------------------------------------------------------------------------
# Ultralytics dataset hook for M1-M5
# ---------------------------------------------------------------------------

def build_yolo_dataset_class(otf: OnTheFlyDataset):
    """
    Build a YOLODataset subclass whose load_image() is routed through
    OnTheFlyDataset for the training split.

    Parameters
    ----------
    otf : OnTheFlyDataset

    Returns
    -------
    type  (subclass of YOLODataset)
    """
    from ultralytics.data.dataset import YOLODataset

    class _AugDataset(YOLODataset):
        """YOLODataset with dynamic augmentation via OnTheFlyDataset."""

        def load_image(self, i, rect_mode=False):
            img_name = Path(self.im_files[i]).name
            otf_idx  = otf.name_to_idx.get(img_name)

            if otf_idx is not None:
                # Training image -- route through augmentation policy
                return otf.load_image(otf_idx)

            # Fallback: val / unmatched -- read straight from disk
            bgr  = cv2.imread(self.im_files[i])
            h, w = bgr.shape[:2]
            return bgr, h, w

    return _AugDataset


# ---------------------------------------------------------------------------
# Shared YAML helper for M1-M5
# ---------------------------------------------------------------------------

def write_data_yaml(
    kitti_yaml_path: Path,
    model_id: str,
    out_dir: Optional[Path] = None,
) -> Path:
    """
    Write a YOLO data YAML for M1-M5 experiments.

    Points train + val at the original KITTI image directories.
    Actual training pixels are intercepted in memory via the
    OnTheFlyDataset hook -- nothing is copied to disk.

    Parameters
    ----------
    kitti_yaml_path : Path
        Original datasets/kitti/kitti.yaml
    model_id : str
        Experiment ID (M1-M5)
    out_dir : Optional[Path]
        Directory to write the generated YAML.
        Defaults to kitti_yaml_path.parent.

    Returns
    -------
    Path
        Path to the generated YAML file.
    """
    kitti_yaml_path = Path(kitti_yaml_path)

    with open(kitti_yaml_path, "r", encoding="utf-8", errors="replace") as f:
        cfg = yaml.safe_load(f)

    kitti_root = kitti_yaml_path.parent

    # Always point at original KITTI dirs
    cfg["train"] = str((kitti_root / "images" / "train").resolve())
    cfg["val"]   = str((kitti_root / "images" / "val").resolve())

    out_dir = Path(out_dir) if out_dir else kitti_yaml_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    out_yaml = out_dir / f"kitti_{model_id}.yaml"

    with open(out_yaml, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)

    return out_yaml
