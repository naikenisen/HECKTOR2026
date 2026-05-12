# HECKTOR2026 - Challenge

<p align="center">
  <img src="HECKTOR_2026_Banner.png">
</p>

Welcome to the **HECKTOR 2026 Challenge** repository! This repository contains instructions and examples for creating a baseline and a valid Docker container for the [HECKTOR 2026 Challenge](TBA). It will also help you understand how to submit your designed model to [Grand Challenge](https://grand-challenge.org/) for evaluation. Here you'll find everything you need to get started quickly: from understanding the challenge, to setting up your environment, training your first model, and evaluating your results. This repository has **two primary branches**:

- [**main**](https://github.com/BioMedIA-MBZUAI/HECKTOR2026/tree/main): Step-by-step guides, data loaders, training scripts, and inference examples so you can get a working model up and running quickly.

- [**docker-template**](https://github.com/BioMedIA-MBZUAI/HECKTOR2026/tree/docker-template): Designed for containerizing and submitting your final models to Grand Challenge. This branch provides a Docker-based inference template, build/test/save scripts, and enforces all challenge restrictions.

---

# How can this Repo help?

1. Understand what the challenge is about
2. Set up your development environment
3. Train models on our provided data
4. Test and evaluate your results
5. Explore ideas for improving performance

---

# 🚀 About the HECKTOR'26 Challenge

Head and Neck (H&N) malignancies constitute a major oncological burden globally, ranking seventh in terms of incidence and occurring more frequently in men and older individuals [Barsouk et al. 2023]. The combination of radiotherapy and cetuximab is currently regarded as a standard therapeutic approach [Bonner et al. 2010]. Nevertheless, disease control at the primary and regional sites remains problematic, with locoregional relapse reported in up to 40% of patients within two years of treatment completion [Chajon et al. 2013]. PET and CT capture distinct yet complementary aspects of tumor biology — metabolic activity and anatomical structure, respectively — providing synergistic information for lesion delineation and for characterizing tumor features that may be predictive of clinical outcomes.

Following the success of previous HECKTOR editions (2020–2025), the **2026 edition introduces a unified, end-to-end pipeline** that jointly addresses segmentation, TN staging, and prognosis for head and neck cancer patients. Unlike prior editions where tasks were treated independently, this framework models the dependency between tasks, closely reflecting real-world clinical decision-making and aligning with the MICCAI 2026 theme of **clinical translation**.

The challenge will be presented at [MICCAI 2026](https://hecktor26.grand-challenge.org). The dataset comprises approximately **1,423 patient cases** from 11+ centers across Canada, Europe, the USA, and the UAE.

---

## The Task: End-to-End Pipeline

There is a **single challenge task** consisting of three sequential, clinically-linked subtasks. All participants must submit results for all three subtasks.

```
FDG-PET/CT + Clinical Data
        │
        ▼
┌───────────────────┐
│  Subtask 1        │  → Segmentation masks (GTVp, GTVn)
│  Segmentation     │     Metric: Mean Dice (GTVp + GTVn)
└────────┬──────────┘
         │  segmentation outputs feed into ▼
┌────────▼──────────┐
│  Subtask 2        │  → T stage + N stage classification
│  TN Staging       │     Metric: Balanced Accuracy + Recall
└────────┬──────────┘
         │  staging outputs feed into ▼
┌────────▼──────────┐
│  Subtask 3        │  → Recurrence-Free Survival (RFS) score
│  Prognosis        │     Metric: C-index
└───────────────────┘
```

Participants may submit **modular approaches** (separately optimized components) or a **single end-to-end model**. End-to-end solutions are encouraged as they more closely reflect real-world clinical scenarios.

### Ranking
The final ranking uses a weighted scheme across subtasks:

| Subtask | Weight | Metric |
|---|---|---|
| Segmentation | 0.25 | Mean Dice (GTVp + GTVn) |
| TN Staging | 0.35 | Mean Balanced Accuracy (T + N) |
| Prognosis | 0.40 | C-index (RFS) |

---

## Challenge Schedule

| Milestone | Date |
|---|---|
| **Training Data Release** | **15 April 2026** |
| Validation Submission Opens | 15 June 2026 |
| Validation Submission Closes | 8 July 2026 |
| Testing Submission Opens | 15 July 2026 |
| Testing Submission Closes | 25 July 2026 |
| Final Report Submission | 8 August 2026 |
| Top 5 Teams Announced | 20 August 2026 |
| Workshop at MICCAI 2026 | 4 or 8 October 2026 |

---

# 📑 Table of Contents

1. [Getting the Data](#-getting-the-data)
2. [Task Folder & Structure](#-task-folder--structure)
3. [Environment Setup](#️-environment-setup)
4. [Training Your Model](#-training-your-model)
5. [Inference & Evaluation](#-inference--evaluation)
6. [Next Steps & Tips](#-next-steps--tips)

---

# 📥 Getting the Data

1. **Download:** Go to the [Data Download Section](https://hecktor26.grand-challenge.org/data-download/) on the challenge website and follow the instructions to download the dataset.

2. **Dataset Structure:** All three subtasks share the same dataset. Each patient folder contains a CT scan, a PET scan, and a segmentation label file. A single CSV provides all clinical data and outcome labels.

```text
hecktor2026_training/
  ├── CHUM-001/
  │   ├── CHUM-001__CT.nii.gz       # CT image
  │   ├── CHUM-001__PT.nii.gz       # PET image (SUV)
  │   └── CHUM-001.nii.gz           # Segmentation label (GTVp=1, GTVn=2)
  ├── CHUM-002/
  ├── ...
  └── HECKTOR_2026_Training.csv     # Clinical data + all outcome labels
```

3. **Dataset Description:** The data originates from FDG-PET and low-dose non-contrast-enhanced CT images of the Head & Neck region, collected from 11+ centers across Canada, Europe, the USA, and the UAE (~1,423 cases total).

- **Image Data (PET/CT):**
  - All cases include paired PET and CT scans using the naming convention: `CenterName_PatientID__Modality.nii.gz`
  - `__CT.nii.gz` — Computed tomography image
  - `__PT.nii.gz` — Positron emission tomography image (standardized uptake values, SUV)

- **Segmentation Labels:**
  - `PatientID.nii.gz` — Label 0 = Background, Label 1 = Primary tumor (GTVp), Label 2 = Lymph nodes (GTVn)
  - If multiple lymph nodes are involved, all share label 2.

- **Clinical Information** (`HECKTOR_2026_Training.csv`):

  | Column | Description |
  |---|---|
  | PatientID | Unique patient identifier |
  | Center | Recruiting institution |
  | Gender | Patient sex |
  | Age | Patient age at scan |
  | Tobacco Consumption | Smoking history |
  | Alcohol Consumption | Alcohol use |
  | Performance Status | ECOG performance status |
  | HPV Status | HPV status (0/1, may be missing) |
  | T_stage | Tumor stage (T1–T4) — **training only** |
  | N_stage | Nodal stage (N0–N3) — **training only** |
  | Relapse | Locoregional recurrence flag — **training only** |
  | RFS | Recurrence-free survival in days — **training only** |

  Some variables may be missing for a subset of patients. TN staging follows the **AJCC/UICC 7th Edition** (N2b and N2c collapsed to N2). The M stage is excluded as the majority of cases are M0 and PET/CT scans are cropped to the H&N region.

---

# 🗂️ Task Folder & Structure

The single challenge task is organized into three subtask folders:

```text
Task/
├── Segmentation/               # Subtask 1: GTVp + GTVn segmentation
│   ├── config/                 # Model configurations
│   ├── data/                   # Dataset and dataloader
│   ├── evaluation/             # Evaluation utilities
│   ├── models/                 # Model architectures (UNet3D, SegResNet, UNETR, SwinUNETR)
│   ├── scripts/                # train.py and inference.py
│   ├── utils/                  # Shared helpers
│   └── README.md               # Subtask-specific documentation
├── TNStaging/                  # Subtask 2: TN staging classification
│   ├── tn_staging.py           # Training script
│   └── tn_staging_inference.py # Inference script
└── Prognosis/                  # Subtask 3: RFS prognosis
    ├── prognosis.py            # Training script
    └── prognosis_inference.py  # Inference script
```

> **Baseline Notice:** This structure and the sample scripts are provided as a **baseline** to help you get started. You are **not required** to follow this exact layout or use the provided models.

---

# ⚙️ Environment Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/BioMedIA-MBZUAI/HECKTOR2026.git
   cd HECKTOR2026
   git checkout main
   ```

2. **Create a virtual environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

---

# 🎯 Training Your Model

Each subtask can be trained independently. The outputs of earlier subtasks can then be used as inputs for later ones.

#### Subtask 1 — Segmentation
```bash
cd Task/Segmentation/
python scripts/train.py --config unet3d
```

#### Subtask 2 — TN Staging
```bash
cd Task/TNStaging/
python tn_staging.py
```

#### Subtask 3 — Prognosis
```bash
cd Task/Prognosis/
python prognosis.py
```

---

# 🔍 Inference & Evaluation

Run inference for each subtask using the commands below. The outputs from Subtask 1 and 2 can be passed as inputs to downstream subtasks.

#### Subtask 1 — Segmentation
```bash
cd Task/Segmentation/
python scripts/inference.py \
    --model_path best_model.pth \
    --ct_path /path/to/ct.nii.gz \
    --pet_path /path/to/pet.nii.gz \
    --output_path /path/to/output
```

#### Subtask 2 — TN Staging
```bash
cd Task/TNStaging/
python tn_staging_inference.py \
    --csv test_data.csv \
    --images_dir ./test_images \
    --checkpoint best_model.pt \
    --output_path /path/to/output.csv
```

#### Subtask 3 — Prognosis
```bash
cd Task/Prognosis/
python prognosis_inference.py \
    --csv test_data.csv \
    --input_path ./test_images \
    --ensemble ensemble_model.pt \
    --clinical_preprocessors hecktor_cache_clinical_preprocessors.pkl
```

---

# 🌟 Next Steps & Tips

* **Data Augmentation:** Explore more aggressive transformations, especially for rare TN stages.
* **End-to-End Training:** Train a single model that jointly optimizes all three subtasks.
* **Feature Propagation:** Pass segmentation-derived features (e.g., tumor volume, SUVmax) into the TN staging and prognosis models.
* **Model Architecture:** Swap in a stronger backbone or use a foundation model.
* **Hyperparameter Tuning:** Adjust learning rates, optimizers, schedulers.
* **Ensembling:** Combine outputs from multiple checkpoints.
* **Class Imbalance:** TN staging has naturally imbalanced class distributions — consider weighted loss or oversampling.

---

# 📚 References

- [Barsouk et al. 2023] Barsouk A, et al. "Epidemiology, Risk Factors, and Prevention of Head and Neck Squamous Cell Carcinoma." Med Sci (Basel). 2023;11(2):42.

- [Bonner et al. 2010] Bonner JA, et al. "Radiotherapy plus Cetuximab for Locoregionally Advanced Head and Neck Cancer: 5-Year Survival Data from a Phase 3 Randomised Trial." The Lancet Oncology 11(1): 21–28.

- [Chajon et al. 2013] Chajon E, et al. "Salivary gland-sparing other than parotid-sparing in definitive head-and-neck intensity-modulated radiotherapy does not seem to jeopardize local control." Radiation Oncology 8.1 (2013): 1–9.

- [Uno et al. 2011] Uno H, et al. "On the C-Statistics for Evaluating Overall Adequacy of Risk Prediction Procedures with Censored Survival Data." Statistics in Medicine 30(10): 1105–17.

---

<div align="center">
  You're now ready to dive in and start building your pipeline!
</div>


# HECKTOR 2026 Challenge - Subtask 1: Segmentation Baselines

A simple and modular framework for training baseline segmentation models for the segmentation subtask of the HECKTOR 2026 Challenge — automatic detection and segmentation of Head and Neck (H&N) primary tumors (GTVp) and lymph nodes (GTVn).

## Overview

This module implements baseline segmentation models for the HECKTOR 2026 Challenge. The segmentation subtask is the first step of the unified end-to-end pipeline: the predicted segmentation masks are used downstream for TN staging and prognosis. The framework is designed to be simple, easy to understand, and easy to extend.

## Features

- **Simple Architecture**: Clean, modular code structure
- **MONAI Integration**: Uses MONAI library for medical image processing
- **Dual Modality**: Supports CT + PET input
- **Comprehensive Evaluation**: Multiple metrics and visualization tools
- **Flexible Configuration**: Easy-to-modify configuration system
- **Ready-to-Use Scripts**: Training, evaluation, and inference scripts

## Installation

1. Clone this repository and navigate to the Segmentation directory:
```bash
git clone https://github.com/BioMedIA-MBZUAI/HECKTOR2026.git
cd HECKTOR2026/Task/Segmentation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Setup

Ensure your HECKTOR 2025 Task 1 data is organized as follows:
```
/path/to/hecktor2025_task1_dataset/
├── imagesTr_cropped/
│   ├── CASE001_0000.nii.gz  # CT images
│   ├── CASE001_0001.nii.gz  # PET images
│   └── ...
└── labelsTr_cropped/
    ├── CASE001.nii.gz       # Tumor and lymph node segmentation masks
    └── ...
```

## Quick Start

### 1. Training

Train a segmentation model for HECKTOR 2025 Task 1 (available models: unet3d, segresnet, unetr, swinunetr):
```bash
python scripts/train.py --config unet3d
```

Training options:
- `--config`: Model configuration (unet3d, segresnet, unetr, swinunetr)
- `--resume`: Path to checkpoint to resume from
- `--device`: Device to use (default: cuda)

### 2. Inference

Run inference on HECKTOR 2025 Task 1 data for a single case:
```bash
python scripts/inference.py \
    --model_path experiments/unet3d/checkpoints/best_model.pth \
    --ct_path /path/to/ct.nii.gz \
    --pet_path /path/to/pet.nii.gz \
    --output_path /path/to/output
```

### 3. Evaluation

For evaluation with ground truth labels, use the `InferenceEvaluator` class programmatically:

```python
from evaluation.inference_evaluator import InferenceEvaluator
from config import UNet3DConfig

# Load config and evaluator
config = UNet3DConfig()
evaluator = InferenceEvaluator(
    model="experiments/unet3d/checkpoints/best_model.pth",
    config=config
)

# Evaluate dataset
results = evaluator.evaluate_dataset(
    data_dir="/path/to/test/data",
    output_dir="evaluation_results"
)
```

The evaluation includes:
- **Dice Score**: Primary segmentation metric
- **Hausdorff Distance**: Surface distance metric
- **IoU**: Intersection over Union
- **Visualization**: Worst case analysis and result plots

## Configuration

Model configurations are defined in the `config/` directory. Available models and configurations:

- **UNet3D**: 3D U-Net with skip connections
- **SegResNet**: Segmentation ResNet architecture
- **UNETR**: Transformer-based U-Net
- **SwinUNETR**: Swin Transformer U-Net

The main parameters include:

- **Data paths**: Location of training data
- **Model architecture**: Network parameters
- **Training settings**: Learning rate, batch size, epochs
- **Augmentation**: Data augmentation parameters

Example configuration (UNet3D):
```python
@dataclass
class UNet3DConfig(BaseConfig):
    # Model
    experiment_name: str = "unet3d"
    channels: tuple = (32, 64, 128, 256, 512)
    
    # Training
    learning_rate: float = 1e-4
    batch_size: int = 2
    num_epochs: int = 200
    
    # Loss
    dice_weight: float = 1.0
    ce_weight: float = 1.0
```

## Model Architecture

### Available Models for HECKTOR 2025 Task 1

**UNet3D (MONAI)**
- **Input**: 3D dual-modality images (CT + PET) for HECKTOR 2025 Task 1
- **Output**: Multi-class segmentation (background + primary tumor + lymph nodes)
- **Architecture**: 3D U-Net with skip connections optimized for head and neck anatomy
- **Features**: 5 encoder/decoder levels with residual connections

**SegResNet**
- **Architecture**: Segmentation ResNet with residual blocks
- **Features**: Deep residual learning for medical image segmentation

**UNETR**
- **Architecture**: Transformer-based U-Net for 3D medical image segmentation
- **Features**: Vision Transformer encoder with CNN decoder

**SwinUNETR**
- **Architecture**: Swin Transformer-based U-Net
- **Features**: Hierarchical vision transformer with U-Net decoder

## Evaluation Metrics

The framework computes the following metrics for HECKTOR 2025 Task 1 evaluation:

- **Dice Score**: Primary segmentation metric for tumor and lymph node regions
- **IoU**: Intersection over Union for overlap assessment
- **Sensitivity/Specificity**: Clinical metrics for detection performance

## Output Structure

After training, the following structure is created:

```
experiments/unet3d/
├── checkpoints/
│   ├── best_model.pth
│   ├── latest_checkpoint.pth
│   └── checkpoint_epoch_*.pth
└── logs/
    ├── metrics.csv
    ├── training_*.log
    └── tensorboard_logs/
```

## Monitoring Training

### TensorBoard
View training progress in real-time:
```bash
tensorboard --logdir experiments/unet3d/logs
```

### Logs
Training logs are saved in `experiments/{model}/logs/training_*.log`

## Data Preprocessing

The framework automatically handles:
- **Intensity normalization**: CT values scaled to [0, 1]
- **Spatial resampling**: Images resized to target size
- **Data augmentation**: Random flips, rotations, scaling
- **Multi-modal stacking**: CT and PET combined as channels

## Extending the Framework

### Adding New Models

1. Create model class in `models/`:
```python
class MyModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        # Initialize your model
    
    def forward(self, x):
        # Implement forward pass
        return output
```

2. Create configuration in `config/`:
```python
@dataclass
class MyModelConfig(BaseConfig):
    experiment_name: str = "my_model"
    # Add model-specific parameters
```

3. Update imports and use in training scripts

### Adding New Loss Functions

Add loss functions to `training/losses.py`:
```python
class MyLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, predictions, targets):
        # Implement loss calculation
        return loss
```

### Adding New Metrics

Add metrics to `training/metrics.py`:
```python
class MyMetric:
    def __call__(self, predictions, targets):
        # Implement metric calculation
        return metric_value
```

## Results

Expected performance on HECKTOR 2025 Task 1 dataset:
- **Dice Score**: ~0.70-0.80 (depending on data quality and tumor complexity)
- **Training time**: ~2-4 hours on modern GPU
- **Memory usage**: ~8-12GB GPU memory

## License

This project is for research purposes related to the HECKTOR 2025 Challenge. Please cite the HECKTOR dataset and challenge if you use this code.

## References

- HECKTOR 2025 Challenge: https://hecktor25.grand-challenge.org/
- MONAI Framework: https://monai.io/
- PyTorch: https://pytorch.org/


# HECKTOR 2026 — Task Overview

The HECKTOR 2026 challenge consists of a **single unified task**: an end-to-end pipeline for head and neck cancer patient assessment using multimodal FDG-PET/CT and clinical data. The pipeline comprises three sequential, clinically-linked subtasks that all participants must complete.

---

## Pipeline Structure

```
FDG-PET/CT + Clinical Data
        │
        ▼
┌───────────────────────────────────┐
│  Subtask 1: Segmentation          │
│  Segment GTVp (primary tumor)     │
│  and GTVn (lymph nodes)           │
│  Metric: Mean Dice (GTVp + GTVn)  │
└────────────────┬──────────────────┘
                 │ segmentation masks + imaging features
                 ▼
┌───────────────────────────────────┐
│  Subtask 2: TN Staging            │
│  Classify T stage (T1–T4) and     │
│  N stage (N0–N3) per AJCC 7th Ed  │
│  Metric: Balanced Accuracy +      │
│          Recall (T and N)         │
└────────────────┬──────────────────┘
                 │ predicted TN stage + imaging features
                 ▼
┌───────────────────────────────────┐
│  Subtask 3: Prognosis             │
│  Predict Recurrence-Free Survival │
│  (RFS) risk score                 │
│  Metric: C-index                  │
└───────────────────────────────────┘
```

---

## Subtask Details

### Subtask 1 — Segmentation (`Segmentation/`)

- **Goal:** Automatically detect and delineate the primary tumor (GTVp) and all metastatic lymph nodes (GTVn) in paired FDG-PET/CT volumes.
- **Output:** A 3D segmentation mask (label 0 = background, 1 = GTVp, 2 = GTVn)
- **Metric:** Mean Dice score averaged over GTVp and GTVn
- **Baseline models:** UNet3D, SegResNet, UNETR, SwinUNETR (via MONAI)
- **Weight in final ranking:** 0.25

### Subtask 2 — TN Staging (`TNStaging/`)

- **Goal:** Classify the radiological T stage (T1–T4) and N stage (N0–N3) for each patient using PET/CT images and clinical information. N subcategories N2b and N2c are collapsed to N2 per AJCC/UICC 7th Edition.
- **Output:** Two categorical predictions per patient: `T_stage` and `N_stage`
- **Metric:** Balanced accuracy and recall for T and N classification
- **Baseline model:** Multimodal ResNet18 with dual classification heads
- **Weight in final ranking:** 0.35

### Subtask 3 — Prognosis (`Prognosis/`)

- **Goal:** Predict each patient's recurrence-free survival (RFS) risk score using PET/CT images, clinical variables, and (optionally) the predicted TN stage from Subtask 2.
- **Output:** A continuous risk score per patient (higher = higher risk)
- **Metric:** Concordance index (C-index)
- **Baseline model:** Cox proportional hazards models
- **Weight in final ranking:** 0.40

---

## Participation Options

Participants may implement:
- **Modular approaches**: separately optimized models for each subtask
- **End-to-end models**: a single model that jointly predicts all three outputs

Both approaches are valid. End-to-end solutions are encouraged as they more closely reflect real-world clinical practice.

---

## Data

All subtasks share the same dataset. See the [main README](../README.md#-getting-the-data) for the full dataset description and download instructions.

The training CSV (`HECKTOR_2026_Training.csv`) includes:
- Clinical variables: age, gender, tobacco, alcohol, performance status, HPV status
- Subtask 1 labels: segmentation masks (`.nii.gz` files)
- Subtask 2 labels: `T_stage`, `N_stage`
- Subtask 3 labels: `Relapse`, `RFS`
