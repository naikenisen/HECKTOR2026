# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

HECKTOR 2026 challenge baseline — head & neck cancer end-to-end pipeline with three chained subtasks: **segmentation → TN staging → prognosis (RFS)**. The `Task/` folder contains baseline implementations; participants are free to replace any component.

Dataset: ~1,423 patients, paired CT + PET `.nii.gz` + `HECKTOR_2026_Training.csv` (clinical data + labels).

## Commands

### Install
```bash
pip install -r requirements.txt
# For GPU: install torch with CUDA first
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Preprocessing
```bash
# Run from repo root — preprocesses raw .nii.gz → resampled/cropped .npz files
python preprocess.py
```

### Training
```bash
# Segmentation (must cd first — uses relative imports)
cd Task/Segmentation/
python scripts/train.py --config swinunetr --fold 0

# TN Staging
cd Task/TNStaging/
python tn_staging.py

# Prognosis
cd Task/Prognosis/
python prognosis.py
```

### Inference
```bash
cd Task/Segmentation/
python scripts/inference.py --model_path best_model.pth --ct_path /path/ct.nii.gz --pet_path /path/pet.nii.gz --output_path /path/output

cd Task/TNStaging/
python tn_staging_inference.py --csv test.csv --images_dir ./images --checkpoint best.pt --scaler_file scaler.pkl --ohe_file ohe.pkl --output_path preds.csv

cd Task/Prognosis/
python prognosis_inference.py --csv test.csv --input_path ./images --ensemble ensemble.pt --clinical_preprocessors preprocessors.pkl
```

## Architecture overview

### Preprocessing (`preprocess.py`)
Operates on raw `.nii.gz` pairs. Pipeline: resample CT+PET to 1mm isotropic → compute intersection bounding box → crop neck region via PET-guided ROI detection (largest high-SUV connected component in top 75% of scan) → MONAI transforms (RAS orientation, CT clipped to [-250, 250] then scaled to [-6, 6], PET z-normalized) → pad to `(310, 200, 200)` → save as `.npz` with image array + metadata.

### Segmentation (`Task/Segmentation/`)
- **Entry points**: `scripts/train.py`, `scripts/inference.py`
- **Config system**: `config/base_config.py` (dataclass) → `config/swinunetr_config.py` inherits it. All paths, hyperparameters, and experiment dirs live here. `__post_init__` creates output directories automatically.
- **Model**: `models/swin_unetr.py` wraps MONAI `SwinUNETR` with `BaseModel` (checkpoint save/load). Input: 2-channel CT+PET patches of `(128,128,128)`.
- **Data pipeline**: `data/transforms.py` loads `.npz` via custom `LoadNpzDictd`, applies `RandCropByLabelClassesd` (ratios `[0.1, 0.45, 0.45]` for bg/GTVp/GTVn), augments, then `ConcatItemsd` merges CT+PET into a single 2-channel image tensor.
- **Training loop**: AdamW + PolynomialLR, DiceCELoss, validation every 5 epochs via sliding window inference (overlap=0.5, gaussian weighting), metric = mean Dice (no background).

### TN Staging (`Task/TNStaging/tn_staging.py`)
Self-contained single file. ResNet-18 3D (FC→Identity, 512-dim) + clinical MLP (→32-dim) → concat 544-dim → T-head + N-head. Input images resized to `96³`. Clinical features: Age (StandardScaler) + Gender/Tobacco/Alcohol/PerformanceStatus/HPV (OneHotEncoder). Loss: `CrossEntropy(T) + CrossEntropy(N)`. 5-fold StratifiedKFold on combined T+N strata.

### Prognosis (`Task/Prognosis/prognosis.py`)
Self-contained single file. `FusedFeatureExtractor`: ResNet-18 3D (FC→Identity, 512-dim) + clinical MLP (→32-dim) → MLP fusion (544→512→256→128-dim) + risk head (128→1). Trained with `DeepHitLoss + 0.1×SurvivalContrastiveLoss`. Then `BaggedIcareSurvival` (from `icare` lib, non-differentiable) is fit on the 128-dim features. Training alternates: (1) update neural net via survival losses, (2) refit BaggedIcareSurvival on new features — repeated for `num_iterations` cycles. Clinical preprocessors are serialized to `.pkl` for inference reuse.

## Key data flow

```
raw .nii.gz → preprocess.py → .npz (resampled+cropped)
                                        │
              ┌─────────────────────────┤
              │                         │
     Segmentation                  TN Staging / Prognosis
     (patch 128³, .npz)            (full image 96³, .nii.gz)
```

Note: segmentation and downstream tasks use **different preprocessing pipelines** — segmentation reads `.npz` files produced by `preprocess.py`; TN staging and prognosis load raw `.nii.gz` directly via MONAI transforms inline.

## Important constraints

- Scripts in `Task/Segmentation/scripts/` must be run **from `Task/Segmentation/`** — they use `sys.path.append(os.path.dirname(os.path.dirname(...)))` to resolve sibling packages (`config`, `models`, `data`, `utils`).
- `Task/TNStaging/` and `Task/Prognosis/` are **self-contained single files** with hardcoded paths (`PATH_TO_TRAINING_IMAGES`, `EHR_DATA_PATH`) at the top that must be updated before running.
- The three subtasks are currently **independent** — no outputs flow between them in the baseline. The challenge encourages chaining them.
- Submission target is Grand Challenge (Docker); the `docker-template` branch contains the containerization template.
- Challenge ranking weights: Segmentation 0.25, TN Staging 0.35, Prognosis 0.40.
