"""Encodeur des données cliniques — MLP simple vers un token d_model."""

import torch
import torch.nn as nn


class ClinicalMLP(nn.Module):
    """
    MLP : (B, n_features) → (B, 64) → (B, d_model) → (B, 1, d_model).

    Hypothèses (cf. README) :
      - variables catégorielles encodées en entiers, NaN → classe "Inconnu"
      - variables continues normalisées, NaN → médiane
      L'imputation est faite en amont dans le dataset.
    """

    def __init__(self, n_features: int = 7, hidden_dim: int = 64, d_model: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, n_features)
        return self.net(x).unsqueeze(1)        # (B, 1, d_model)
