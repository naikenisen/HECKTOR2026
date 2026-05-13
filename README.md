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
  CT+PET (RAW)
        |
        | preprocessing (src/preprocessing.py)
        |
CT+PET (B, 2, 128, 128, 128)
        │
        ▼
    SwinUNETR (src/models.py)
        │
    L_Seg = Dice+Focal (utils/losses.py)
        |
        | segmentation & embeddings (src/segmentation_emb.py)
        |
        ├──► seg_mask (B, 3, D, H, W)
        │
        └──► bottleneck (B, C, D', H', W')
                  │
        ┌─────────┼─────────┐
        │                   │
        ▼                   ▼
T-Head (models.py)    N-Head (models.py)
  (GAP + Linear)      (GAP + Linear)
        │                   |
        |                   |
        |                   │
        ▼                   ▼
   Logits T (B,4)     Logits N (B,4)  
        │                   │
   L_T = CrossEnt     L_N = CrossEnt  (utils/losses.py)
        │                   │
        └─────────┬─────────┘
                  │
                  |   tn classification and embeddings  (src/tn_staging_emb.py)
                  |
         nn.Linear(8, d_model)
         token_tn (B, 1, d_model)
                  │
                  │          Clinical (B, 7)
                  │               │
                  │     MLP (7→64→d_model) (models.py)
                  |               |
                  |               | features enrichement (src/features_enrichement.py)
                  │               │
                  ▼               ▼
        ┌─────────────────────────────────────┐
        │         Cross-Attention Fusion      │
        │         (models/cross_attention.py) │
        │                                     │
        │  Q (B, 3, d_model) :                │
        │  cat([CLS, token_clin, token_tn])   │
        │                                     │
        │  KV (B, N+M, d_model) :             │
        │  cat([bottleneck flatten,           │
        │       seg_mask flatten])            │
        │                                     │
        │  → MultiheadAttention(Q, KV, KV)    │
        │  → CLS token enrichi (B, d_model)   │
        └─────────────────  ──────────────────┘
                          |
                          ▼
                Survival Head nn.Linear(d_model → 256 → T) (src/models.py)
                          │
                          |  survival prediction (src/survival.py)
                          |
                logits bruts (B, T)
                          |
                          | L_Surv = DeepHit (utils/losses/py)
                          │
                      softmax(logits)
                          | 
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
