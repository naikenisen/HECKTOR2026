"""SegResNet model using MONAI, optionally initialized from vista3d.pt weights."""

import os
import torch
import torch.nn as nn
from monai.networks.nets import SegResNet


class SegResNetModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.segresnet = SegResNet(
            spatial_dims=config.spatial_dims,
            init_filters=config.init_filters,
            in_channels=config.input_channels,
            out_channels=config.num_classes,
            dropout_prob=config.dropout_prob,
            blocks_down=config.blocks_down,
            blocks_up=config.blocks_up,
            upsample_mode=config.upsample_mode,
        )

        if config.pretrained_path and os.path.exists(config.pretrained_path):
            self._load_pretrained(config.pretrained_path)
        elif config.pretrained_path:
            print(f"[SegResNet] Pretrained weights not found at '{config.pretrained_path}' — training from scratch.")

    def _load_pretrained(self, path: str):
        """
        Load vista3d.pt weights with strict=False.
        convInit (in_channels mismatch 1→2) and conv_final (out_channels mismatch)
        are skipped and stay randomly initialized.
        """
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)

        if isinstance(checkpoint, dict):
            for key in ("state_dict", "model", "model_state_dict", "net"):
                if key in checkpoint:
                    checkpoint = checkpoint[key]
                    break

        def strip_prefix(sd, prefix):
            return {(k[len(prefix):] if k.startswith(prefix) else k): v for k, v in sd.items()}

        for prefix in ("module.", "segresnet.", "model."):
            if any(k.startswith(prefix) for k in checkpoint):
                checkpoint = strip_prefix(checkpoint, prefix)

        missing, unexpected = self.segresnet.load_state_dict(checkpoint, strict=False)

        interior_missing = [k for k in missing if "convInit" not in k and "conv_final" not in k]
        if interior_missing:
            print(f"[SegResNet] Warning — missing interior keys: {interior_missing}")

        print(
            f"[SegResNet] Loaded pretrained weights from '{path}'. "
            f"Skipped (channel mismatch): convInit, conv_final. "
            f"Unexpected keys: {len(unexpected)}."
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.segresnet(x)

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
