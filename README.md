# AugScale: YOLOv8 Augmentation & Robustness Evaluation Framework

A comprehensive framework for training YOLOv8 object detection models with different augmentation strategies and evaluating their robustness across diverse conditions.

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Training Models](#training-models)
  - [Evaluating Models](#evaluating-models)
  - [Analysis & Visualization](#analysis--visualization)
- [Datasets](#datasets)
- [Models](#models)
- [Results](#results)
- [Development](#development)
- [Requirements](#requirements)

---

## Overview

ScaleVision is designed to systematically evaluate how different augmentation strategies impact YOLOv8 model robustness. The framework trains 6 model variants (M0-M5) with varying augmentation policies and evaluates each under 37 different conditions to measure degradation and resilience.

### Pipeline

```
Train Models (M0-M5) → Evaluate (37 conditions) → Rank by Robustness → Visualize Results
```

**Key Objectives:**
- Train baseline and augmented YOLOv8 models with controlled experiments
- Evaluate robustness under weather, blur, noise, illumination, and geometric variations
- Rank models by robustness metrics
- Generate comprehensive visualizations and analysis reports

---

## Project Structure

```
ScaleVision/
├── configs/                    # Configuration files
│   ├── train.yaml             # Training hyperparameters & model definitions
│   ├── augmentations.yaml     # Augmentation policies (baseline, geometric, weather, etc.)
│   └── evaluation.yaml        # Evaluation condition definitions
│
├── datasets/                  # Dataset storage
│   ├── kitti/                 # KITTI object detection dataset
│   │   ├── images/
│   │   └── labels/
│   ├── kitti_aug/             # Augmented KITTI variants
│   └── brain_tumor/           # Brain tumor detection dataset
│
├── experiments/               # Experimental results
│   ├── M0/                    # Baseline model (no augmentation)
│   ├── M1/                    # Geometric augmentation
│   ├── M2/                    # Weather augmentation
│   ├── M3/                    # Blur/Noise augmentation
│   ├── M4/                    # Illumination augmentation
│   ├── M5/                    # Mixed robust augmentation
│   │   ├── args.yaml          # Model training arguments
│   │   ├── results.csv        # Per-epoch training metrics
│   │   ├── weights/           # Model checkpoints
│   │   │   ├── best.pt
│   │   │   ├── last.pt
│   │   │   └── epoch*.pt
│   │   └── eval/              # Evaluation results
│   │       ├── raw_eval.csv   # Per-condition metrics
│   │       └── cond_*/        # YOLO validation outputs per condition
│   └── ...
│
├── models/                    # Pre-trained model weights
│   ├── yolov8n.pt            # YOLOv8 nano
│   └── yolov8m.pt            # YOLOv8 medium
│
├── results/                   # Aggregated results & visualizations
│   ├── raw_eval.csv          # Aggregated metrics (all models)
│   ├── robustness_ranking.csv # Robustness rankings
│   ├── heatmaps/             # Sensitivity heatmaps
│   ├── degradation_curves/   # Performance degradation plots
│   └── model_comparison.png  # Comparative visualization
│
├── scripts/                   # CLI entry points
│   ├── train.py              # Train models (M0-M5)
│   ├── evaluate.py           # Evaluate robustness (37 conditions)
│   ├── run_phase2.py         # Complete pipeline orchestration
│   └── verify_setup.py       # Environment verification
│
├── src/                       # Core implementation modules
│   ├── training/
│   │   ├── trainer.py        # YOLOv8 training pipeline
│   │   ├── dataset.py        # Dataset handling & augmentation
│   │   └── config.py         # Configuration loading & management
│   │
│   ├── evaluation/
│   │   ├── evaluator.py      # Robustness evaluation engine
│   │   ├── conditions.py     # 37 evaluation conditions
│   │   ├── metrics.py        # Performance metrics computation
│   │   ├── ranking.py        # Model robustness ranking
│   │   └── plots.py          # Visualization generation
│   │
│   ├── augmentations/
│   │   ├── policies.py       # Augmentation policy definitions
│   │   ├── registry.py       # Policy registry & lookup
│   │   ├── transforms.py     # Individual augmentation transforms
│   │   └── __init__.py
│   │
│   └── utils/
│       ├── io.py             # File I/O utilities
│       ├── logger.py         # Logging configuration
│       ├── seed.py           # Reproducibility (random seeds)
│       └── __init__.py
│
├── notebook/                 # Jupyter analysis notebooks
│   ├── data_analysis.ipynb   # Dataset exploration
│   ├── augmentation.ipynb    # Augmentation visualization
│   └── comparison.ipynb      # Model comparison analysis
│
├── visuals/                  # Visual assets & analysis outputs
│   ├── analysis/             # Analysis figures
│   └── augments/             # Augmentation examples
│
├── requirements.txt          # Python dependencies (see Installation)
└── README.md                # This file
```

---

## Features

### Training Capabilities
- **Multiple Augmentation Strategies**: 6 predefined policies (baseline, geometric, weather, blur/noise, illumination, mixed)
- **Controlled Experimentation**: Consistent hyperparameters across all models, only augmentation varies
- **YOLOv8 Integration**: Uses Ultralytics YOLOv8 (nano, medium variants)
- **Dynamic Augmentation**: In-memory augmentation during training (M1-M5) vs. static dataset (M0)
- **Checkpoint Management**: Saves weights at every epoch with best-model tracking

### Evaluation Features
- **37 Robustness Conditions**: Weather, blur, noise, illumination, and geometric variations
- **Per-Condition Metrics**: Confidence, precision, recall, mAP50, mAP50-95
- **Aggregated Analysis**: Cross-model comparison and robustness ranking
- **Reproducibility**: Controlled random seeds, deterministic augmentation application

### Analysis & Visualization
- **Sensitivity Heatmaps**: Performance degradation by condition
- **Degradation Curves**: Model robustness trajectories
- **Model Comparison**: Side-by-side performance visualization
- **Ranking Reports**: Robustness scores and percentile rankings
- **Jupyter Notebooks**: Interactive exploration and analysis

---

## Installation

### Prerequisites
- Python 3.10
- pip (or conda)
- CUDA 13.0+ (for GPU support)
- NVIDIA GPU recommended (RTX 4090 or similar for optimal performance)

### Step-by-Step Installation

**1. Upgrade pip & build tools:**
```bash
pip install --upgrade pip setuptools wheel
```

**2. Install PyTorch with CUDA 13.0 support:**
```bash
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu130
```

**3. Install project dependencies:**
```bash
pip install -r requirements.txt
```

**4. Verify GPU setup (optional but recommended):**
```bash
python -c "import torch; print(torch.cuda.get_device_name(0), torch.version.cuda)"
# Expected: NVIDIA GeForce RTX 5060 Ti  13.0
```

**5. Verify project setup:**
```bash
python scripts/verify_setup.py
```

---

## Quick Start

### Train All Models (M0-M5)
```bash
python scripts/train.py --model all
```

### Train a Single Model
```bash
python scripts/train.py --model M1
```

### Evaluate Models (37 conditions)
```bash
python scripts/evaluate.py --model all
```

### Evaluate Single Model
```bash
python scripts/evaluate.py --model M2
```

### Run Complete Phase-2 Pipeline
```bash
python scripts/run_phase2.py
```

### View Results
- **Aggregated metrics**: `results/raw_eval.csv`
- **Robustness ranking**: `results/robustness_ranking.csv`
- **Visualizations**: `results/heatmaps/`, `results/degradation_curves/`
- **Per-model details**: `experiments/<model_id>/eval/raw_eval.csv`

---

## Configuration

### Training Configuration (`configs/train.yaml`)

```yaml
# YOLO variant to train
yolo_variant: "models/yolov8m.pt"

# Models to train/evaluate
active_models: ["M0", "M1", "M2", "M3", "M4", "M5"]

models:
  M0: { policy: baseline,     description: "No augmentation" }
  M1: { policy: geometric,    description: "Family A — Geometric" }
  M2: { policy: weather,      description: "Family B — Weather" }
  M3: { policy: blur_noise,   description: "Family C — Blur/Noise" }
  M4: { policy: illumination, description: "Family D — Illumination" }
  M5: { policy: mixed,        description: "Mixed Robust (all families)" }
```

### Augmentation Policies (`configs/augmentations.yaml`)

Six predefined policies control which transforms are applied during training:

1. **baseline**: No augmentation (M0)
2. **geometric**: Rotation, scaling, shear, affine transforms (M1)
3. **weather**: Rain, fog, snow, sun glare (M2)
4. **blur_noise**: Gaussian blur, motion blur, noise (M3)
5. **illumination**: Brightness, contrast, gamma, saturation (M4)
6. **mixed**: Combined from all families (M5)

### Evaluation Configuration (`configs/evaluation.yaml`)

Defines 37 conditions across 5 categories:
- **Weather** (9 conditions): rain, fog, snow, etc.
- **Blur** (4 conditions): motion blur, defocus, etc.
- **Noise** (6 conditions): Gaussian, Poisson, salt-pepper, etc.
- **Illumination** (10 conditions): brightness, contrast variations
- **Geometric** (8 conditions): rotation, scaling, perspective, etc.

### Override Configuration

Pass command-line overrides:
```bash
python scripts/train.py --model M1 --epochs 30 --device cpu --cfg-override batch_size=16
```

---

## Usage

### Training Models

#### Train All Active Models
```bash
python scripts/train.py --model all
```
Trains M0-M5 sequentially according to `configs/train.yaml`.

#### Train Specific Model
```bash
python scripts/train.py --model M2
```

#### Training with Hyperparameter Overrides
```bash
python scripts/train.py --model M1 --epochs 50 --batch_size 32 --device gpu
```

#### List Available Augmentation Policies
```bash
python scripts/train.py --list-policies
```

### Evaluating Models

#### Evaluate All Models
```bash
python scripts/evaluate.py --model all
```
Runs 37-condition evaluation on each of M0-M5.

#### Evaluate Specific Model
```bash
python scripts/evaluate.py --model M3
```

#### Ranking-Only Mode
If raw_eval.csv already exists, compute rankings without re-evaluating:
```bash
python scripts/evaluate.py --rank-only
```

#### Full Pipeline with Plotting
```bash
python scripts/evaluate.py --model all --plot
```
Generates heatmaps, degradation curves, and model comparison visualization.

### Analysis & Visualization

#### Open Jupyter Notebooks
```bash
jupyter notebook notebook/
```

**Available Notebooks:**
- `data_analysis.ipynb` — Dataset exploration, class distributions
- `augmentation.ipynb` — Visualization of augmentation effects
- `comparison.ipynb` — Model robustness comparison analysis

#### Generate Visualizations
```bash
python scripts/evaluate.py --model all --plot
```

**Outputs:**
- `results/heatmaps/` — Condition sensitivity heatmaps
- `results/degradation_curves/` — Performance degradation plots
- `results/model_comparison.png` — Comparative visualization

---

## Datasets

### KITTI Dataset
- **Location**: `datasets/kitti/`
- **Task**: Object detection (cars, pedestrians, cyclists)
- **Split**: Train / Validation
- **Format**: Images + YOLO-format labels
- **Usage**: Primary dataset for M0-M5 training

### Brain Tumor Dataset (Optional)
- **Location**: `datasets/brain_tumor/`
- **Task**: Tumor detection/segmentation
- **Split**: Train / Validation
- **Format**: Images + YOLO-format labels
- **Status**: Available for alternative experiments

### Dataset Preparation for M0
Baseline model (M0) uses a static copy of KITTI on disk:
```bash
python scripts/train.py --model M0
# Copies original KITTI → experiments/M0/dataset/
# Ultralytics loads from disk (standard pipeline)
```

### Dynamic Augmentation (M1-M5)
Augmented models apply transforms in-memory during training:
```bash
python scripts/train.py --model M1
# Original KITTI remains untouched
# Augmentations applied per-sample during loading
# Validation data kept clean for fair comparison
```

---

## Models

### Supported YOLO Variants
- **yolov8n** (nano) — Lightweight, ~3M parameters
- **yolov8m** (medium) — Default, ~25M parameters

Located in `models/` and referenced via `configs/train.yaml`.

### Model Checkpoints
Each trained model stores:
```
experiments/<model_id>/weights/
├── best.pt          # Best validation mAP
├── last.pt          # Final epoch
├── epoch0.pt        # First epoch
├── epoch10.pt       # Every 10th epoch
└── ...
```

### Model Arguments
Training configuration saved per model:
```
experiments/<model_id>/args.yaml
```

---

## Results

### Output Structure

All results aggregated to `results/`:

```
results/
├── raw_eval.csv                 # All metrics, all conditions, all models
├── robustness_ranking.csv       # Ranked by robustness score
├── heatmaps/
│   ├── M0_sensitivity.png
│   ├── M1_sensitivity.png
│   └── ...
├── degradation_curves/
│   ├── M0_degradation.png
│   ├── M1_degradation.png
│   └── ...
└── model_comparison.png         # Overall comparison
```

### Key Metrics
- **Confidence** — Average prediction confidence
- **Precision** — TP / (TP + FP)
- **Recall** — TP / (TP + FN)
- **mAP50** — Mean Average Precision (IoU 0.50)
- **mAP50-95** — Mean Average Precision (IoU 0.50-0.95)
- **Robustness Score** — Aggregated resilience metric

### Per-Model Results
```
experiments/<model_id>/eval/
├── raw_eval.csv               # Per-condition metrics
```

---

## Development

### Module Overview

#### Training (`src/training/`)
- `trainer.py` — Main training loop, checkpoint management
- `dataset.py` — Dataset loading, augmentation application
- `config.py` — Configuration parsing and validation

#### Evaluation (`src/evaluation/`)
- `evaluator.py` — Robustness evaluation engine
- `conditions.py` — 37 evaluation condition implementations
- `metrics.py` — Metric computation (confidence, precision, recall, mAP, etc.)
- `ranking.py` — Robustness ranking algorithm
- `plots.py` — Visualization generation

#### Augmentations (`src/augmentations/`)
- `policies.py` — Augmentation policy definitions
- `registry.py` — Policy lookup and management
- `transforms.py` — Individual transform implementations

#### Utilities (`src/utils/`)
- `io.py` — File I/O, CSV writing, path management
- `logger.py` — Logging setup
- `seed.py` — Random seed initialization (reproducibility)

### Adding New Augmentation Policies

1. Define transforms in `src/augmentations/transforms.py`
2. Create policy in `src/augmentations/policies.py`
3. Register in `src/augmentations/registry.py`
4. Add to `configs/augmentations.yaml`
5. Update `configs/train.yaml` if needed

### Adding New Evaluation Conditions

1. Implement condition in `src/evaluation/conditions.py`
2. Register in condition registry
3. Update `configs/evaluation.yaml`
4. Re-run evaluation pipeline

### Running Tests
```bash
pytest tests/
```

---

## Requirements

### Python Environment
- **Python**: 3.10
- **CUDA**: 13.0+ (optional, for GPU)

### Key Dependencies
See [Installation](#installation) for step-by-step setup.

**Core Packages:**
- **torch** == 2.11.0
- **torchvision** == 0.26.0
- **ultralytics** (YOLOv8)
- **albumentations** (augmentation)
- **scikit-image**, **scikit-learn**, **numpy**, **pandas**
- **matplotlib**, **seaborn** (visualization)
- **jupyter**, **ipython** (notebooks)

### Full Requirements
See `requirements.txt` for complete dependency list with pinned versions.

---

## Troubleshooting

### GPU Not Detected
```bash
python -c "import torch; print(torch.cuda.is_available())"
# If False, re-install PyTorch with CUDA 13.0 support (see Installation Step 2)
```

### Out of Memory (OOM)
Reduce batch size:
```bash
python scripts/train.py --model M1 --batch_size 16
```

### Missing Dataset
Ensure KITTI dataset exists at `datasets/kitti/`:
- `datasets/kitti/images/train/` and `datasets/kitti/images/val/`
- `datasets/kitti/labels/train/` and `datasets/kitti/labels/val/`

### Import Errors
Verify project is in Python path:
```bash
python scripts/verify_setup.py
```

---

## Citation & References

If you use ScaleVision in your research, please cite:

```bibtex
@project{scalevision2026,
  title={ScaleVision: YOLOv8 Augmentation \& Robustness Evaluation Framework},
  author={Your Organization},
  year={2026},
  url={https://github.com/your-org/scalevision}
}
```

---

## License

[Your License Here]

---

## Contact & Support

For questions, issues, or contributions:
- **Issues**: [GitHub Issues](https://github.com/your-org/scalevision/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/scalevision/discussions)

---

**Last Updated**: May 2026  
**Status**: Active Development  
**Maintainers**: ScaleVision Team
