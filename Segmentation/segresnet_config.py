"""SegResNet configuration — vista3d.pt pretrained backbone."""

import os
from dataclasses import dataclass
from typing import Tuple


@dataclass
class SegResNetConfig:

    # Data paths
    data_root: str = "/path/to/hecktor2026_training"
    train_images_dir: str = "imagesTr_resampled_cropped_npy"
    train_labels_dir: str = "labelsTr_resampled_cropped_npy"

    # Data properties
    input_channels: int = 2                          # CT + PET
    num_classes: int = 3                             # bg + GTVp + GTVn
    spatial_size: Tuple[int, int, int] = (128, 128, 128)

    # Training
    batch_size: int = 2
    learning_rate: float = 1e-2
    weight_decay: float = 3e-5
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
    experiment_name: str = "segresnet"
    output_dir: str = "experiments"

    # Architecture — must match vista3d.pt checkpoint
    spatial_dims: int = 3
    init_filters: int = 32          # vista3d uses 32
    blocks_down: tuple = (1, 2, 2, 4)
    blocks_up: tuple = (1, 1, 1)
    dropout_prob: float = 0.2
    upsample_mode: str = "nontrainable"

    # Pretrained weights (set to None to train from scratch)
    pretrained_path: str = "vista3d.pt"

    def __post_init__(self):
        self.experiment_dir = os.path.join(self.output_dir, self.experiment_name)
        self.checkpoint_dir = os.path.join(self.experiment_dir, "checkpoints")
        self.log_dir        = os.path.join(self.experiment_dir, "logs")
        for d in [self.experiment_dir, self.checkpoint_dir, self.log_dir]:
            os.makedirs(d, exist_ok=True)
