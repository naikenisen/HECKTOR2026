"""MultitaskModel — connecte tous les composants et garantit la backprop end-to-end."""

import torch
import torch.nn as nn

from src.swinunetr import SwinUNETRConfig, SwinUNETRMultitask
from src.heads import TNHead, SurvivalHead
from src.clinical_encoder import ClinicalMLP
from src.cross_attention import CrossAttentionFusion


class MultitaskModel(nn.Module):
    """
    Pipeline HECKTOR 2026 end-to-end :

      CT+PET → SwinUNETR → (seg_mask, bottleneck)
      bottleneck → T-Head  → (t_feat, t_logits)
      bottleneck → N-Head  → (n_feat, n_logits)
      [t_feat || n_feat] → token_tn
      Clinical → ClinicalMLP → token_clin
      [CLS, token_clin, token_tn] →(cross-attn vs bottleneck aplati) → cls_out
      cls_out → SurvivalHead → surv_logits

    Tous les gradients (Seg, T, N, Survie) remontent jusqu'au backbone.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # ---- 1. Backbone ----
        sw_cfg = SwinUNETRConfig(
            input_channels=config.input_channels,
            num_classes=config.num_seg_classes,
            spatial_size=config.spatial_size,
            feature_size=config.feature_size,
            use_checkpoint=config.use_checkpoint,
            pretrained_path=config.pretrained_path,
        )
        self.backbone = SwinUNETRMultitask(sw_cfg)

        # ---- 2. Têtes T / N ----
        C_bottleneck = config.bottleneck_channels  # 768 par défaut pour feature_size=48
        self.t_head = TNHead(C_bottleneck, hidden_dim=config.hidden_tn,
                             num_classes=config.num_t_classes)
        self.n_head = TNHead(C_bottleneck, hidden_dim=config.hidden_tn,
                             num_classes=config.num_n_classes)

        # ---- 3. Tokens ----
        self.proj_tn   = nn.Linear(2 * config.hidden_tn, config.d_model)
        self.clin_mlp  = ClinicalMLP(n_features=config.n_clinical_features,
                                     hidden_dim=64, d_model=config.d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.d_model) * 0.02)

        # ---- 4. Fusion ----
        self.cross_attn = CrossAttentionFusion(
            bottleneck_channels=C_bottleneck,
            d_model=config.d_model,
            n_heads=config.n_heads,
        )

        # ---- 5. Tête survie ----
        self.surv_head = SurvivalHead(
            d_model=config.d_model,
            hidden_dim=config.surv_hidden,
            n_time_bins=config.n_time_bins,
        )

    def forward(self, ct_pet: torch.Tensor, clinical: torch.Tensor) -> dict:
        # 1. Backbone
        seg_mask, bottleneck = self.backbone(ct_pet)

        # 2. T / N — features riches + logits
        t_feat, t_logits = self.t_head(bottleneck)             # (B, hidden), (B, 4)
        n_feat, n_logits = self.n_head(bottleneck)             # (B, hidden), (B, 4)

        # 3. Token TN à partir des features
        token_tn = self.proj_tn(torch.cat([t_feat, n_feat], dim=1)).unsqueeze(1)
        # (B, 1, d_model)

        # 4. Token clinique
        token_clin = self.clin_mlp(clinical)                   # (B, 1, d_model)

        # 5. CLS expandé au batch
        B = ct_pet.size(0)
        cls = self.cls_token.expand(B, -1, -1).contiguous()    # (B, 1, d_model)

        # 6. Queries concaténées
        Q = torch.cat([cls, token_clin, token_tn], dim=1)      # (B, 3, d_model)

        # 7. Cross-attention vs bottleneck
        cls_out = self.cross_attn(Q, bottleneck)               # (B, d_model)

        # 8. Survie (logits bruts ; softmax à l'inférence)
        surv_logits = self.surv_head(cls_out)                  # (B, T)

        return {
            "seg_mask":    seg_mask,
            "t_logits":    t_logits,
            "n_logits":    n_logits,
            "surv_logits": surv_logits,
        }

    def save_checkpoint(self, path: str, epoch: int, optimizer_state=None, **kwargs):
        ckpt = {
            "epoch": epoch,
            "model_state_dict": self.state_dict(),
            **kwargs,
        }
        if optimizer_state is not None:
            ckpt["optimizer_state_dict"] = optimizer_state
        torch.save(ckpt, path)

    def load_checkpoint(self, path: str, device: str = "cpu") -> dict:
        ckpt = torch.load(path, map_location=device)
        self.load_state_dict(ckpt["model_state_dict"])
        return ckpt
