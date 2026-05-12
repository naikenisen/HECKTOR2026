"""Base configuration class for all models."""

import os
from dataclasses import dataclass
from typing import Tuple


@dataclass
class BaseConfig:
    """Base configuration class with common parameters."""
    
    # Data paths
    data_root: str = "/path/to/hecktor2026_training"
    train_images_dir: str = "imagesTr_resampled_cropped_npy"
    train_labels_dir: str = "labelsTr_resampled_cropped_npy"
    # Data properties
    input_channels: int = 2  # CT + PET
    num_classes: int = 3     # background + primary tumor + metastatic tumor
    spatial_size: Tuple[int, int, int] = (128, 128, 128)
    
    # Training parameters
    batch_size: int = 2
    learning_rate: float = 1e-2
    weight_decay: float = 3e-5
    num_epochs: int = 350
    
    # Scheduler parameters
    # PolyLR scheduler parameters
    poly_lr_power: float = 0.9
    poly_lr_min_lr: float = 1e-6
    
    # Data augmentation
    use_augmentation: bool = True
    aug_probability: float = 0.5
    rotation_range: float = 15.0
    scaling_range: float = 0.1
    translation_range: float = 10.0
    
    # System parameters
    device: str = "cuda"
    num_workers: int = 4
    pin_memory: bool = True
    
    # Data caching parameters
    cache_rate: float = 0.25  # Cache 25% of training data for faster loading
    
    # Checkpointing and logging
    save_checkpoint_every: int = 1 # Save checkpoint every n epochs
    use_tensorboard: bool = True
    
    # Output directories
    experiment_name: str = "baseline"
    output_dir: str = "experiments"

    def __post_init__(self):
        """Setup output directories."""
        self.experiment_dir = os.path.join(self.output_dir, self.experiment_name)
        self.checkpoint_dir = os.path.join(self.experiment_dir, "checkpoints")
        self.log_dir = os.path.join(self.experiment_dir, "logs")

        for dir_path in [self.experiment_dir, self.checkpoint_dir, self.log_dir]:
            os.makedirs(dir_path, exist_ok=True)
