"""Fusion par cross-attention CLS / Clinique / TN → bottleneck spatial."""

import torch
import torch.nn as nn


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention multi-tête.

      Q : (B, 3, d_model)  = [CLS, token_clin, token_tn]
      K = V : (B, N, d_model) = Linear(C, d_model)(bottleneck.flatten(2).permute(0,2,1))

    Renvoie le token CLS enrichi : (B, d_model).
    """

    def __init__(self, bottleneck_channels: int, d_model: int = 256, n_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.kv_proj = nn.Linear(bottleneck_channels, d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm_ffn = nn.LayerNorm(d_model)

    def forward(self, queries: torch.Tensor, bottleneck: torch.Tensor) -> torch.Tensor:
        # queries: (B, 3, d_model)
        # bottleneck: (B, C, D', H', W')
        B, C = bottleneck.shape[:2]
        kv = bottleneck.flatten(2).permute(0, 2, 1)        # (B, N, C)
        kv = self.kv_proj(kv)                              # (B, N, d_model)
        kv = self.norm_kv(kv)

        q = self.norm_q(queries)                           # (B, 3, d_model)
        attn_out, _ = self.attn(q, kv, kv, need_weights=False)
        x = queries + attn_out                             # résidu
        x = x + self.ffn(self.norm_ffn(x))                 # FFN + résidu

        # On renvoie uniquement le CLS enrichi (premier token)
        return x[:, 0]                                     # (B, d_model)
