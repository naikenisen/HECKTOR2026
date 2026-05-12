# Pipeline actuelle — HECKTOR 2026 (analyse détaillée)

Les trois tâches sont **entièrement indépendantes** : aucune sortie d'une tâche n'alimente une autre.

---

## Tâche 1 — Segmentation tumorale

### Modèle : SwinUNETR (MONAI)
Architecture Transformer hiérarchique de type encoder-decoder. L'encodeur est un Swin Transformer 3D (attention par fenêtres glissantes), le décodeur est un UNet classique avec skip connections.

### Entrée
- CT + PET **concaténés comme 2 canaux** avant d'entrer dans le réseau
- Format : `.npz` préprocessés, chargés à pleine résolution
- Aucun resize global — des **patches aléatoires** de taille `(128, 128, 128)` sont extraits à la volée via `RandCropByLabelClassesd` avec ratios `[0.1, 0.45, 0.45]` (fond / tumeur primaire / ganglions) pour équilibrer les classes
- À l'inférence : **sliding window** de taille `(128, 128, 128)` avec overlap 0.5 et pondération gaussienne

### Augmentations (entraînement uniquement)
- Flip aléatoire (axes 0, 1, 2)
- Variation d'intensité CT (scale ±10%, shift ±10%)
- Bruit gaussien CT (std=0.01)
- Lissage gaussien CT (σ ∈ [0.5, 1.15])

### Paramètres SwinUNETR
| Paramètre | Valeur |
|-----------|--------|
| `img_size` | (128, 128, 128) |
| `in_channels` | 2 (CT + PET) |
| `out_channels` | 3 (fond + tumeur + ganglions) |
| `feature_size` | 48 |
| `depths` | (2, 2, 2, 2) |
| `num_heads` | (3, 6, 12, 24) |
| `drop_rate` | 0.0 |
| `attn_drop_rate` | 0.0 |
| `dropout_path_rate` | 0.0 |

### Loss
**DiceCELoss** (MONAI) — combinaison de Dice Loss et Cross-Entropy Loss :
- `to_onehot_y=True` : les labels sont convertis en one-hot
- `softmax=True` : softmax appliqué sur les prédictions
- `include_background=True` : la classe fond est incluse dans le calcul

### Optimisation
| Paramètre | Valeur |
|-----------|--------|
| Optimiseur | AdamW |
| Learning rate | 1e-2 |
| Weight decay | 3e-5 |
| Scheduler | PolynomialLR (power=0.9, min_lr=1e-6) |
| Epochs | 350 |
| Batch size | 2 |
| Cache rate | 25% des données en mémoire |

### Métrique de validation
**Dice moyen** (sans fond) — calculé toutes les 5 époques via sliding window inference.

### Ce sur quoi le modèle est entraîné
Segmentation supervisée voxel par voxel sur les **masques de segmentation manuels** fournis par le challenge (3 classes).

---

## Tâche 2 — TN Staging

### Modèle : ResNet-18 3D + MLP clinique
Deux branches indépendantes dont les sorties sont **concaténées**, suivies de deux têtes de classification.

```
CT + PET (2 canaux, 96³)
        │
   ResNet-18 3D
   (FC → Identity)
        │
   f_img : 512 dims
        │
        ├────────────────────────────────┐
                                         │
Données cliniques                        │
(Age, Gender, Tobacco,                   │
 Alcohol, Perf. Status,                  │
 HPV Status)                             │
        │                                │
   MLP clinique                          │
   64 → ReLU → 32 → ReLU                 │
        │                                │
   f_clin : 32 dims                      │
        │                                │
        └──── Concaténation ─────────────┘
                    │
               544 dims
              /         \
         T-head         N-head
       (544→128→T)    (544→128→N)
              │               │
          T-stage          N-stage
```

### Entrée images
- CT + PET empilés comme **2 canaux**
- **Redimensionnés à 96³** (perte de résolution importante par rapport aux images originales)
- Transforms : LoadImage → EnsureChannelFirst → ScaleIntensity → Resize → ToTensor

### Entrée clinique
| Feature | Type | Traitement |
|---------|------|-----------|
| Age | Numérique | StandardScaler |
| Gender | Catégoriel | OneHotEncoder |
| Tobacco Consumption | Catégoriel | OneHotEncoder |
| Alcohol Consumption | Catégoriel | OneHotEncoder |
| Performance Status | Catégoriel | OneHotEncoder |
| HPV Status | Catégoriel | OneHotEncoder |

### Sorties
- **T-stage** : classification 4 classes (T1, T2, T3, T4)
- **N-stage** : classification 4 classes (N0, N1, N2, N3)

### Loss
**CrossEntropyLoss** pour T + **CrossEntropyLoss** pour N, sommées :
```python
loss = criterion(t_logits, t_lbl) + criterion(n_logits, n_lbl)
```

### Optimisation
| Paramètre | Valeur |
|-----------|--------|
| Optimiseur | Adam |
| Learning rate | 1e-4 |
| Epochs | 10 |
| Batch size | 4 |
| Validation | BalancedAccuracy (T et N) |

### Validation croisée
- 5-fold StratifiedKFold stratifié sur la combinaison T_stage + N_stage
- Split test holdout 20% avant la CV

### Ce sur quoi le modèle est entraîné
Classification supervisée sur les **labels T-stage et N-stage** fournis dans le CSV du challenge. Le ResNet-18 n'a aucun préentraînement — il apprend à extraire des features utiles uniquement via le signal de classification T/N.

### Limitation principale
Le réseau voit l'image entière à **96³** sans aucune information sur la localisation de la tumeur — il doit la localiser implicitement dans une image dégradée.

---

## Tâche 3 — Pronostic de survie (RFS)

### Modèle : FusedFeatureExtractor + BaggedIcareSurvival
Pipeline en deux étapes : (1) un réseau neuronal extrait des features, (2) un modèle de survie classique prédit le risque.

```
CT + PET (2 canaux, 96³)
        │
   ResNet-18 3D
   (FC → Identity)
        │
   f_img : 512 dims
        │
        ├────────────────────────────────────┐
                                             │
Données cliniques                            │
(Age, Gender, Tobacco,                       │
 Alcohol, Perf. Status,                      │
 M-stage, Treatment)                         │
        │                                    │
   MLP clinique                              │
   clin_dim → 64 → BN → Dropout(0.3)        │
            → 64 → BN → Dropout(0.2)        │
            → 32 → ReLU                      │
        │                                    │
   f_clin : 32 dims                          │
        │                                    │
        └──── Concaténation ─────────────────┘
                    │
               544 dims
                    │
          MLP de fusion
          544 → 512 → BN → Dropout(0.4)
              → 256 → BN → Dropout(0.3)
              → 128
                    │
           features : 128 dims
          /                    \
    Risk head               BaggedIcareSurvival
    128→64→1                 (entraîné sur 128 dims)
         │                           │
  risk_score              score de risque RFS final
  (pour les losses)
```

### Entrée images
- CT + PET empilés comme **2 canaux**
- **Redimensionnés à 96³**
- Transforms : LoadImage → EnsureChannelFirst → ScaleIntensity → Resize → ToTensor

### Entrée clinique
| Feature | Type | Traitement |
|---------|------|-----------|
| Age | Numérique | StandardScaler (fit sur train) |
| Gender | Catégoriel | OneHotEncoder |
| Tobacco Consumption | Catégoriel | OneHotEncoder |
| Alcohol Consumption | Catégoriel | OneHotEncoder |
| Performance Status | Catégoriel | OneHotEncoder |
| M-stage | Catégoriel | OneHotEncoder |
| Treatment | Catégoriel | OneHotEncoder |

### Losses (deux losses combinées)

**1. DeepHitLoss** — appliquée sur les `risk_scores` de la risk head :
- Composante **likelihood** : log-vraisemblance du modèle de Cox (qui survit le plus longtemps parmi ceux à risque ?)
- Composante **ranking** : les patients avec événement précoce doivent avoir un score de risque plus élevé que ceux qui survivent plus longtemps
- `ranking_weight=0.3`, `ranking_scale=1.0`

**2. SurvivalContrastiveLoss** — appliquée sur les features 128 dims :
- Paires similaires (temps de survie proches + événement) → features proches
- Paires dissimilaires (grands écarts de survie) → features éloignées
- `margin=2.0`, `temperature=0.1`

**Loss totale** :
```python
total_loss = DeepHitLoss(risk_scores) + 0.1 * SurvivalContrastiveLoss(features)
```

### Optimisation du FusedFeatureExtractor
| Paramètre | Valeur |
|-----------|--------|
| Optimiseur | Adam |
| Learning rate | 1e-3 |
| Weight decay | 1e-5 |
| Scheduler | ReduceLROnPlateau (mode=max, factor=0.5, patience=3) |
| Gradient clipping | max_norm=1.0 |

### Entraînement itératif (alternance)
Le système utilise une **optimisation alternée** sur `num_iterations` cycles :
1. Entraîner le FusedFeatureExtractor pendant `feature_epochs_per_iteration` époques (via DeepHitLoss + ContrastiveLoss)
2. Réentraîner le BaggedIcareSurvival sur les nouvelles features extraites
3. Évaluer le C-index sur la validation → sauvegarder le meilleur état

Par défaut : 25 itérations × 10 époques = 250 époques effectives.

### BaggedIcareSurvival
- Modèle de survie de la librairie `icare`
- `aggregation_method='median'`
- `n_jobs=-1` (parallélisme)
- Entraîné sur les vecteurs **128 dims** extraits par le FusedFeatureExtractor
- Ne fait pas de backpropagation — c'est un modèle classique (non différentiable)

### Métrique
**C-index** (concordance index) — mesure la capacité du modèle à ordonner correctement les patients par risque de récidive.

### Ce sur quoi le modèle est entraîné
- **FusedFeatureExtractor** : entraîné sur les labels de survie RFS (temps + événement) via DeepHitLoss et SurvivalContrastiveLoss
- **BaggedIcareSurvival** : entraîné en mode supervisé classique sur les features 128 dims + labels de survie structurés `[(event, time), ...]`

### Limitation principale
CT et PET sont fusionnés **naïvement dès l'entrée** (2 canaux), à **96³** de résolution, sans aucune information sur la localisation de la tumeur et sans lien avec les résultats du staging T/N.

---

## Résumé global

| | Segmentation | TN Staging | Pronostic |
|---|---|---|---|
| **Backbone** | SwinUNETR | ResNet-18 3D | ResNet-18 3D |
| **Résolution entrée** | Patches 128³ (pleine résolution) | 96³ (dégradée) | 96³ (dégradée) |
| **Fusion CT/PET** | 2 canaux dès l'entrée | 2 canaux dès l'entrée | 2 canaux dès l'entrée |
| **Données cliniques** | ✗ | ✓ (concat après encodage) | ✓ (concat après encodage) |
| **Loss** | DiceCELoss | CrossEntropy (T) + CrossEntropy (N) | DeepHitLoss + 0.1×ContrastiveLoss |
| **Modèle de prédiction final** | Softmax voxel-wise | T-head / N-head | BaggedIcareSurvival |
| **Métrique** | Dice (sans fond) | Balanced Accuracy | C-index |
| **Lien avec les autres tâches** | ✗ | ✗ | ✗ |
