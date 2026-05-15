"""
src/training/__init__.py
=========================
Public training package interface.

Exports
-------
TrainConfig         Immutable shared experiment configuration
get_train_config    Factory for TrainConfig objects
OnTheFlyDataset     Dynamic in-memory augmentation dataset (M1-M5)
write_data_yaml     YOLO dataset YAML generator (M1-M5)
write_m0_yaml       Direct KITTI YAML passthrough for M0 (no copying)
ModelTrainer        End-to-end experiment training pipeline
"""

from .config import TrainConfig, get_train_config
from .dataset import OnTheFlyDataset, write_data_yaml, write_m0_yaml
from .trainer import ModelTrainer
