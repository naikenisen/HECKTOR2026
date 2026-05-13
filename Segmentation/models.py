"""SwinUNETR model using MONAI, initialized from SSL pretrained weights."""

import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR


class SwinUNETRModel(nn.Module):

    def __init__(self, config):
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
