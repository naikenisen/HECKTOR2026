"""Métriques d'évaluation : Dice, Balanced Accuracy, C-index."""

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score
from lifelines.utils import concordance_index


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Balanced Accuracy pour staging T ou N (entiers)."""
    mask = y_true >= 0
    if mask.sum() == 0:
        return 0.0
    return float(balanced_accuracy_score(y_true[mask], y_pred[mask]))


def discrete_risk_from_surv_logits(surv_logits: torch.Tensor) -> torch.Tensor:
    """
    Convertit (B, T) logits → score de risque scalaire (B,).
    On utilise l'espérance pondérée 1 − CIF, ou plus simplement la somme softmax pondérée
    par le rang temporel : un risque élevé = événement précoce attendu.
    """
    p = torch.softmax(surv_logits, dim=-1)              # (B, T)
    T = surv_logits.size(-1)
    bins = torch.arange(T, device=surv_logits.device, dtype=p.dtype) + 1.0
    # Plus la masse est sur des bins précoces, plus le risque est élevé →
    # risk = -E[bin]  (signe inversé pour que ↑ risk ↔ ↓ temps)
    expected_bin = (p * bins).sum(dim=-1)               # (B,)
    return -expected_bin


def c_index(risk_scores: np.ndarray, times: np.ndarray, events: np.ndarray) -> float:
    """C-index. risk_scores : score élevé = risque élevé (donc temps court attendu)."""
    if events.sum() == 0:
        return 0.5
    # lifelines : concordance_index(times, predicted_scores, event_observed)
    # avec predicted_scores tel que scores élevés ↔ longue durée. Donc -risk.
    return float(concordance_index(times, -risk_scores, events))
