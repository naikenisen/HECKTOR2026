"""SwinUNETR — configuration et modèle."""

import os
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Tuple
from monai.networks.nets import SwinUNETR


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

    # MONAI SSL pretrained SwinViT encoder (required)
    # Download: https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/model_swinvit.pt
    pretrained_path: str = "model_swinvit.pt"

    def __post_init__(self):
        self.experiment_dir = os.path.join(self.output_dir, self.experiment_name)
        self.checkpoint_dir = os.path.join(self.experiment_dir, "checkpoints")
        self.log_dir        = os.path.join(self.experiment_dir, "logs")
        for d in [self.experiment_dir, self.checkpoint_dir, self.log_dir]:
            os.makedirs(d, exist_ok=True)


class SwinUNETRModel(nn.Module):

    def __init__(self, config: SwinUNETRConfig):
        super().__init__()
        self.config = config

        self.swinunetr = SwinUNETR(
            img_size=config.spatial_size,
            in_channels=config.input_channels,
            out_channels=config.num_classes,
            feature_size=config.feature_size,
            use_checkpoint=config.use_checkpoint,
        )

        weights = torch.load(config.pretrained_path, map_location="cpu", weights_only=False)
        if "state_dict" in weights:
            weights = weights["state_dict"]
        self.swinunetr.load_from(weights=weights)
        print(f"[SwinUNETR] Loaded pretrained encoder from '{config.pretrained_path}'. Decoder initialised randomly.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.swinunetr(x)

    def save_checkpoint(self, path: str, epoch: int, optimizer_state: dict = None, **kwargs):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.state_dict(),
            "model_config": self.config.__dict__,
            **kwargs,
        }
        if optimizer_state:
            checkpoint["optimizer_state_dict"] = optimizer_state
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str, device: str = "cpu") -> dict:
        checkpoint = torch.load(path, map_location=device)
        self.load_state_dict(checkpoint["model_state_dict"])
        return checkpoint

    def get_parameters(self) -> dict:
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total_parameters":     total,
            "trainable_parameters": trainable,
            "model_size_mb":        total * 4 / (1024 ** 2),
        }
