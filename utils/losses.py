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

# Survival — DeepHit discret (NLL + ranking sur la CIF)
class DeepHitDiscreteLoss(nn.Module):
    """
    DeepHit discret pour survie en temps discrétisé.

    Inputs
    ------
    logits       : (B, T)            logits bruts du survival head
    times        : (B,)              temps de suivi (en années / mois — peu importe)
    events       : (B,)              indicateur d'événement (1=relapse, 0=censuré)
    bin_edges    : (T+1,)            bornes des intervalles temporels (quantiles fit sur train)

    Loss = L_NLL + α · L_rank
      L_NLL  : -log p(bin_event | x) pour les événements observés
               -log P(T > bin_censor | x) pour les censurés
      L_rank : pénalise un classement incorrect du CIF entre paires comparables
    """

    def __init__(self, alpha: float = 0.2, sigma: float = 0.1):
        super().__init__()
        self.alpha = alpha
        self.sigma = sigma

    @staticmethod
    def time_to_bin(times: torch.Tensor, bin_edges: torch.Tensor) -> torch.Tensor:
        """Retourne l'index du bin (0..T-1) pour chaque temps."""
        T = bin_edges.numel() - 1
        # torch.bucketize : retourne l'index de l'intervalle (1..T), on clamp → 0..T-1
        idx = torch.bucketize(times, bin_edges[1:-1], right=False)
        return idx.clamp(0, T - 1).long()

    def forward(self, logits: torch.Tensor, times: torch.Tensor,
                events: torch.Tensor, bin_edges: torch.Tensor) -> torch.Tensor:
        device = logits.device
        B, T = logits.shape
        pmf = torch.softmax(logits, dim=-1)                          # (B, T)
        cif = torch.cumsum(pmf, dim=-1)                              # (B, T)

        k = self.time_to_bin(times, bin_edges)                       # (B,)
        events_b = events.bool()

        # ---- NLL ----
        eps = 1e-8
        p_event = pmf.gather(1, k.unsqueeze(1)).squeeze(1)           # (B,)
        # Survie au-delà du bin k pour les censurés : 1 - CIF[k]
        s_censor = 1.0 - cif.gather(1, k.unsqueeze(1)).squeeze(1)    # (B,)

        nll_event  = -torch.log(p_event[events_b].clamp_min(eps)).sum() \
                     if events_b.any() else torch.tensor(0.0, device=device)
        nll_censor = -torch.log(s_censor[~events_b].clamp_min(eps)).sum() \
                     if (~events_b).any() else torch.tensor(0.0, device=device)
        l_nll = (nll_event + nll_censor) / max(B, 1)

        # ---- Ranking ----
        # Paires (i, j) tel que événement_i = 1 et t_i < t_j (j peut être censuré).
        # On veut CIF_i(k_i) > CIF_j(k_i) → on pénalise sinon.
        if events_b.any():
            cif_at_ki = cif.gather(1, k.unsqueeze(1))                # (B, 1)
            # cif_i_at_ki : (B,) ; cif_j_at_ki : besoin (B, B) via index k_i pour chaque j
            #   cif_j_at_ki[i, j] = cif[j, k_i]
            ki_expand = k.view(1, B).expand(B, B)                    # (B_i, B_j) chacun = k_i
            cif_j_at_ki = cif.unsqueeze(0).expand(B, B, T).gather(2, ki_expand.unsqueeze(-1)).squeeze(-1)
            #   .unsqueeze(0) : (1, B, T)  →  expand (B_i, B_j, T)
            cif_i_at_ki = cif_at_ki.expand(B, B)                     # (B_i, B_j)

            time_mat  = times.view(B, 1) < times.view(1, B)          # i a un temps < j
            valid     = events_b.view(B, 1) & time_mat               # (B_i, B_j)

            diff = (cif_i_at_ki - cif_j_at_ki)                       # > 0 si bien classé
            rank_loss = torch.exp(-diff / self.sigma)
            rank_loss = (rank_loss * valid.float()).sum() / (valid.float().sum() + eps)
        else:
            rank_loss = torch.tensor(0.0, device=device)

        return l_nll + self.alpha * rank_loss

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
