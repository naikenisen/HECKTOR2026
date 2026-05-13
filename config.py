"""Configuration globale pour la pipeline 2026 multitâche end-to-end."""

import os
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class MultitaskConfig:
    # ---------- Données ----------
    data_root: str = "../../hecktor2026_training"
    train_images_dir: str = "imagesTr_resampled_cropped_npy"
    train_labels_dir: str = "labelsTr_resampled_cropped_npy"
    csv_path: str = "../../hecktor2026_training/HECKTOR_2026_Training.csv"

    # Modalités / classes
    input_channels: int = 2                  # CT + PET
    num_seg_classes: int = 3                 # bg + GTVp + GTVn
    num_t_classes: int = 4                   # T1..T4
    num_n_classes: int = 4                   # N0..N3
    spatial_size: Tuple[int, int, int] = (128, 128, 128)

    # ---------- Backbone (SwinUNETR + SSL) ----------
    feature_size: int = 48
    use_checkpoint: bool = True
    pretrained_path: str = "model_swinvit.pt"
    # Pour SwinUNETR feature_size=48 + input 128³ → bottleneck = (B, 768, 4, 4, 4)
    bottleneck_channels: int = 768

    # ---------- Fusion ----------
    d_model: int = 256
    n_heads: int = 4
    n_clinical_features: int = 7             # cf. README : MLP(7 → 64 → d_model)
    hidden_tn: int = 256                     # hidden dim des T/N heads

    # ---------- Survie discrète ----------
    n_time_bins: int = 10                    # T intervalles définis par quantiles
    surv_hidden: int = 256

    # ---------- Entraînement ----------
    batch_size: int = 2
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    num_epochs: int = 350
    n_warmup: int = 20                       # epochs warm-up (seg seule)
    grad_clip_norm: float = 1.0
    poly_lr_power: float = 0.9

    # Augmentation
    use_augmentation: bool = True
    aug_probability: float = 0.5

    # Split
    val_split: float = 0.2
    seed: int = 42

    # Système
    device: str = "cuda"
    num_workers: int = 4
    cache_rate: float = 0.25
    save_checkpoint_every: int = 5
    use_tensorboard: bool = True

    # Sortie
    experiment_name: str = "multitask_e2e"
    output_dir: str = "experiments"

    def __post_init__(self):
        self.experiment_dir = os.path.join(self.output_dir, self.experiment_name)
        self.checkpoint_dir = os.path.join(self.experiment_dir, "checkpoints")
        self.log_dir        = os.path.join(self.experiment_dir, "logs")
        for d in [self.experiment_dir, self.checkpoint_dir, self.log_dir]:
            os.makedirs(d, exist_ok=True)
