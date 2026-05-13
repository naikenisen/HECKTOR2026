"""SwinUNETR configuration — MONAI SSL pretrained encoder."""

import os
from dataclasses import dataclass
from typing import Tuple


@dataclass
class SwinUNETRConfig:

    # Data paths
    data_root: str = "/path/to/hecktor2026_training"
    train_images_dir: str = "imagesTr_resampled_cropped_npy"
    train_labels_dir: str = "labelsTr_resampled_cropped_npy"

    # Data properties
    input_channels: int = 2                           # CT + PET
    num_classes: int = 3                              # bg + GTVp + GTVn
    spatial_size: Tuple[int, int, int] = (128, 128, 128)

    # Training
    batch_size: int = 2
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    num_epochs: int = 350
    poly_lr_power: float = 0.9
    poly_lr_min_lr: float = 1e-6

    # Augmentation
    use_augmentation: bool = True
    aug_probability: float = 0.5

    # System
    device: str = "cuda"
    num_workers: int = 4
    cache_rate: float = 0.25
    save_checkpoint_every: int = 1
    use_tensorboard: bool = True

    # Output
    experiment_name: str = "swinunetr"
    output_dir: str = "experiments"

    # Architecture — must match pretrained checkpoint (feature_size=48 for MONAI SSL)
    feature_size: int = 48
    use_checkpoint: bool = True   # gradient checkpointing — saves VRAM during training

    # Pretrained weights — MONAI SSL pretrained SwinViT encoder
    # Download: https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/model_swinvit.pt
    pretrained_path: str = "model_swinvit.pt"

    def __post_init__(self):
        self.experiment_dir = os.path.join(self.output_dir, self.experiment_name)
        self.checkpoint_dir = os.path.join(self.experiment_dir, "checkpoints")
        self.log_dir        = os.path.join(self.experiment_dir, "logs")
        for d in [self.experiment_dir, self.checkpoint_dir, self.log_dir]:
            os.makedirs(d, exist_ok=True)
