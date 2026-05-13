"""Têtes (heads) — staging T/N et survie discrète."""

import torch
import torch.nn as nn


class TNHead(nn.Module):
    """
    Tête de staging (T ou N).
    Reçoit le bottleneck (B, C, D', H', W'), applique un Global Average Pooling
    puis renvoie à la fois la feature intermédiaire (B, hidden) et les logits (B, num_classes).
    La feature riche est utilisée pour la fusion cross-attention.
    """

    def __init__(self, in_channels: int, hidden_dim: int = 256, num_classes: int = 4):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.feat = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, bottleneck: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # bottleneck: (B, C, D', H', W')
        x = self.gap(bottleneck).flatten(1)     # (B, C)
        feat = self.feat(x)                     # (B, hidden_dim)
        logits = self.classifier(feat)          # (B, num_classes)
        return feat, logits


class SurvivalHead(nn.Module):
    """
    Tête de survie discrète (DeepHit-style).
    Reçoit le token CLS enrichi par cross-attention (B, d_model)
    et produit des logits bruts (B, T) sur T intervalles temporels.
    Le softmax n'est appliqué qu'à l'inférence.
    """

    def __init__(self, d_model: int, hidden_dim: int = 256, n_time_bins: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, n_time_bins),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
