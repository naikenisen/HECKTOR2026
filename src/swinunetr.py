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


class SwinUNETRMultitask(nn.Module):
    """
    SwinUNETR exposant à la fois :
      - le masque de segmentation final (B, num_classes, D, H, W)
      - la feature bottleneck du SwinViT (B, C, D', H', W')  — feature la plus profonde

    On réutilise le SwinUNETR de MONAI : son attribut `swinViT` renvoie la liste
    des feature maps hiérarchiques [stage0, stage1, stage2, stage3, stage4].
    Le bottleneck correspond au dernier élément (downscale ×32, C=feature_size*16).
    """

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

        if config.pretrained_path and os.path.exists(config.pretrained_path):
            weights = torch.load(config.pretrained_path, map_location="cpu", weights_only=False)
            if "state_dict" in weights:
                weights = weights["state_dict"]
            self.swinunetr.load_from(weights=weights)
            print(f"[SwinUNETRMultitask] SSL weights chargés depuis '{config.pretrained_path}'.")
        else:
            print(f"[SwinUNETRMultitask] ATTENTION : poids SSL introuvables ({config.pretrained_path}). Décodeur init aléatoire.")

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Reproduit le forward de monai.networks.nets.SwinUNETR mais expose en plus
        le bottleneck. Cf. https://github.com/Project-MONAI/MONAI swinunetr.py.
        """
        net = self.swinunetr
        hidden_states_out = net.swinViT(x, net.normalize)
        # hidden_states_out = [stage0, stage1, stage2, stage3, bottleneck]
        bottleneck = hidden_states_out[4]

        enc0 = net.encoder1(x)
        enc1 = net.encoder2(hidden_states_out[0])
        enc2 = net.encoder3(hidden_states_out[1])
        enc3 = net.encoder4(hidden_states_out[2])
        dec4 = net.encoder10(hidden_states_out[4])
        dec3 = net.decoder5(dec4, hidden_states_out[3])
        dec2 = net.decoder4(dec3, enc3)
        dec1 = net.decoder3(dec2, enc2)
        dec0 = net.decoder2(dec1, enc1)
        out  = net.decoder1(dec0, enc0)
        seg_logits = net.out(out)

        return seg_logits, bottleneck
