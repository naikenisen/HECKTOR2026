# HECKTOR 2026 

## Aproches précédentes
- Les trois tâches du challenge (segmentation, staging, pronostic) sont **complètement isolées** : aucune information ne circule entre elles.
- L'image est dégradée à 96³, sans information sur la localisation tumorale.
- Pas d'utilisation des masques de segmentation pour le pronostic.
---

## Consignes 2026
Participants are invited to develop a multimodal pipeline leveraging FDG PET, CT, and clinical data to:

    Segment primary tumors and lymph nodes
    Infer radiological TN staging
    Predict recurrence-free survival

This unified task reflects a realistic clinical workflow, integrating diagnosis, staging, and prognosis into a single framework.

## Pipeline 2026 pour la prédiction de la survie par End-to-End Multitask Learning

```
CT+PET (2 canaux, 128³)
        │
        ▼
┌─────────────────────────────────┐
|     SwinUNETR - pretrained      │
└─────────────────┬───────────────┘
                  │
                  ▼
         Bottleneck (Latent z)
                  │
  ┌───────────────┼───────────────┐
  │               │               │
  ▼               ▼               ▼
Décodeur       T-Head          N-Head
  |          (GAP+Linear)    (GAP+Linear)
  │               │               │
  ▼               ▼               ▼
Masque          Logits T        Logits N
(B,1,D,H,W)    (B, 4)          (B, 4)
  │               │               │
  │  L_Seg        │  L_T          │  L_N
  │  (Dice+Focal) │  (CrossEnt)   │  (CrossEnt)
  │               │               │
  │               └───────┬───────┘
  │                       │
  │               Projection linéaire
  │               nn.Linear(8, d_model)
  │               token_tn (B, 1, d_model)
  │                       │                    ┌──────────────────────────┐
  │                       │                    │ Données Cliniques (B, 7) │
  │                       │                    │ Âge, Sexe, HPV, M-stage..│
  │                       │                    └────────────┬─────────────┘
  │                       │                                 │
  │                       │                    Tabular Transformer
  │                       │                    (embedding par feature
  │                       │                     + self-attention)
  │                       │                    token_clin (B, 1, d_model)
  │                       │                                 │
  │                       ▼                                 ▼
  │        ┌──────────────────────────────────────────────────────────┐
  │        │              Cross-Attention Fusion                      │
  │        │                                                          │
  │        │  Q (B, 3, d_model) :                                     │
  │        │  torch.cat([CLS token, token_clin, token_tn], dim=1)     │
  │        │                                                          │
  │        │  K / V (B, N+M, d_model) :                               │
  │        │  torch.cat([Bottleneck z (flatten), Masque (flatten)])   │
  │        │                                                          │
  │        │  → nn.MultiheadAttention(Q, KV, KV)                      │
  │        │  → CLS token enrichi (B, d_model)                        │
  └───────►│                                                          │
           └──────────────────────────┬───────────────────────────────┘
                                      │
                                      ▼
                        Survival-Head (Discrete-Time)
                         nn.Linear(d_model→256→T)
                                      │
                                      ▼
                           logits bruts (B, T)
                                      │   L_Surv (DeepHit)
                                 softmax(logits)
                                      │
                                      ▼
                          Risk Probabilities (B, T)
═══════════════════════════════════════════════════════════════════════════════

FONCTION DE PERTE TOTALE (End-to-End) :

  L_Total = w₁·L_Seg + w₂·L_T + w₃·L_N + w₄·L_Surv

  • Poids dynamiques (wᵢ) ajustés automatiquement par Uncertainty Weighting
  • Les gradients de toutes les pertes remontent jusqu'au Bottleneck
  • Rétropropagation de L_Surv via la cross-attention → Bottleneck
  • Warm-up segmentation avant entraînement multitâche complet

═══════════════════════════════════════════════════════════════════════════════

SORTIES CLINIQUES — Rapport généré par patient

  ┌─────────────────────────────────────────────────────────────────┐
  │                    RAPPORT PATIENT                              │
  │                                                                 │
  │  1. SEGMENTATION                                                │
  │     • Masque tumoral 3D (tumeur primaire + ganglions)           │
  │     • Visualisation superposée sur CT/PET                       │
  │     • Volume tumoral (cm³), volume ganglionnaire (cm³)          │
  │                                                                 │
  │  2. STAGING TN                                                  │
  │     • T-stage prédit : T1 / T2 / T3 / T4                        │
  │       avec probabilités : [0.05, 0.72, 0.18, 0.05]              │
  │     • N-stage prédit : N0 / N1 / N2 / N3                        │
  │       avec probabilités : [0.10, 0.65, 0.20, 0.05]              │
  │                                                                 │
  │  3. PRONOSTIC DE SURVIE                                         │
  │     • C-index (évaluation de la discrimination)                 │
  │     • Courbe de risque sur T intervalles de temps :             │
  │                                                                 │
  │       Risque de récidive (%)                                    │
  │       40% │         ╭──────                                     │
  │       30% │      ╭──╯                                           │
  │       20% │   ╭──╯                                              │
  │       10% │╭──╯                                                 │
  │        0% └──────────────────────── Temps                       │
  │              6m  12m  18m  24m  36m                             │
  │                                                                 │
  │     • Probabilité cumulée de récidive à 1 an : XX%              │
  │     • Probabilité cumulée de récidive à 2 ans : XX%             │
  └─────────────────────────────────────────────────────────────────┘
```