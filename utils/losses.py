"""Loss functions for the HECKTOR 2026 multitask pipeline."""

import torch
import torch.nn as nn
from monai.losses import DiceFocalLoss

# Segmentation — MONAI DiceFocalLoss
seg_loss = DiceFocalLoss(
    to_onehot_y=True,
    softmax=True,
    include_background=False,
    reduction="mean",
)


# TN Staging — standard CrossEntropy (one per head)
t_loss = nn.CrossEntropyLoss()
n_loss = nn.CrossEntropyLoss()

# Survival — DeepHit (Cox likelihood + pairwise ranking)
_EPSILON = 1e-8


class DeepHitLoss(nn.Module):
    """DeepHit-style loss: negative partial log-likelihood + ranking term."""

    def __init__(self, ranking_weight: float = 0.2, ranking_scale: float = 1.0):
        super().__init__()
        self.ranking_weight = ranking_weight
        self.ranking_scale  = ranking_scale

    def forward(self, risk_scores: torch.Tensor,
                survival_times: torch.Tensor,
                event_indicators: torch.Tensor) -> torch.Tensor:

        if event_indicators.sum() == 0:
            return torch.tensor(0.01, device=risk_scores.device, requires_grad=True)

        device     = risk_scores.device
        batch_size = len(survival_times)

        # Likelihood
        sorted_idx    = torch.argsort(survival_times, descending=True)
        sorted_scores = risk_scores[sorted_idx]
        sorted_events = event_indicators[sorted_idx]

        ll = torch.tensor(0.0, device=device, requires_grad=True)
        for i in range(batch_size):
            if sorted_events[i] == 1:
                ll = ll + sorted_scores[i] - torch.logsumexp(sorted_scores[i:], dim=0)
        ll = -ll / (event_indicators.sum() + _EPSILON)

        # Ranking
        rl, pairs = torch.tensor(0.0, device=device, requires_grad=True), 0
        for i in range(batch_size):
            for j in range(i + 1, batch_size):
                if event_indicators[i] == 1 and survival_times[i] < survival_times[j]:
                    rl = rl + torch.exp(self.ranking_scale * (risk_scores[j] - risk_scores[i]))
                    pairs += 1
                elif event_indicators[j] == 1 and survival_times[j] < survival_times[i]:
                    rl = rl + torch.exp(self.ranking_scale * (risk_scores[i] - risk_scores[j]))
                    pairs += 1
        if pairs > 0:
            rl = rl / pairs

        return ll + self.ranking_weight * rl


surv_loss = DeepHitLoss(ranking_weight=0.2)

# Uncertainty Weighting — Kendall et al. 2018
class UncertaintyWeightedLoss(nn.Module):
    """
    Combines N task losses with learnable uncertainty weights.
    Add the parameters of this module to the optimizer alongside the model.
    """

    def __init__(self, n_tasks: int = 4):
        super().__init__()
        # log(σ) initialised to 0  →  σ=1, weight=0.5 at the start
        self.log_sigma = nn.Parameter(torch.zeros(n_tasks))

    def forward(self, *task_losses: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            *task_losses: individual scalar losses (L_seg, L_T, L_N, L_surv)
        Returns:
            total_loss, weights  (weights = 1/(2σᵢ²) for logging)
        """
        assert len(task_losses) == self.log_sigma.numel(), \
            f"Expected {self.log_sigma.numel()} losses, got {len(task_losses)}"

        sigma_sq = torch.exp(2 * self.log_sigma)           # σᵢ²
        weights  = 1.0 / (2.0 * sigma_sq)                  # 1/(2σᵢ²)

        total = sum(w * l + s for w, l, s in
                    zip(weights, task_losses, self.log_sigma))
        return total, weights.detach()
