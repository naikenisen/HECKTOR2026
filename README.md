# HECKTOR 2026 

## Aproches précédentes
- Les trois tâches du challenge (segmentation, staging, pronostic) sont **complètement isolées** : aucune information ne circule entre elles.
- L'image est dégradée à 96³, sans information sur la localisation tumorale.
- Pas d'utilisation des masques de segmentation pour le pronostic.

## Consignes 2026
Participants are invited to develop a multimodal pipeline leveraging FDG PET, CT, and clinical data to:

    Segment primary tumors and lymph nodes
    Infer radiological TN staging
    Predict recurrence-free survival

This unified task reflects a realistic clinical workflow, integrating diagnosis, staging, and prognosis into a single framework.

## Pipeline 2026 pour la prédiction de la survie par End-to-End Multitask Learning

```markdown
  CT+PET (RAW)
        |
        | preprocessing (src/preprocessing.py)
        |
CT+PET (B, 2, 128, 128, 128)
        |
        | dataloading and transformation (src/dataloader.py and src/transforms.py)
        │
        ▼
    SwinUNETR (src/swinunetr.py)
        │
    L_Seg = Dice+Focal (utils/losses.py)
        |
        | segmentation & embeddings (src/model.py)
        |
        ├──► seg_mask (B, 3, D, H, W)
        │
        └──► bottleneck (B, C, D', H', W')
                  │
        ┌─────────┼─────────┐
        │                   │
        ▼                   ▼
T-Head (heads.py)    N-Head (heads.py)
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
                  |   tn classification and embeddings  (src/model.py)
                  |
         nn.Linear(8, d_model)
         token_tn (B, 1, d_model)
                  │
                  │          Clinical (B, 7)
                  │               │
                  │     MLP (7→64→d_model) (clinical_encoder.py)
                  |               |
                  |               | features enrichement (src/model.py)
                  │               │
                  ▼               ▼
      ┌─────────────────────────────────────────┐
      │  Q : token_clin, token_tn, CLS          │
      │  K = V : bottleneck (image)             │ cross-attention fusion (src/cross_attention.py)
      │                                         │
      │  attn(Q, K, V) (src/models.py)          │
      └─────────────────────────────────────────┘
                          |
                          ▼
                Survival Head nn.Linear(d_model → 256 → T) (src/heads.py)
                          │
                          |  survival prediction (src/model.py)
                          |
                logits bruts (B, T)
                          |
                          | L_Surv = DeepHit (utils/losses.py)
                          │
                      softmax(logits)
                          | 
                          ▼
                  Risk Probabilities (B, T)
═══════════════════════════════════════════════════════════════════════════════
Un script train.py permet de faire tourner le training

FONCTION DE PERTE TOTALE (End-to-End) :
  L_Total = w₁·L_Seg + w₂·L_T + w₃·L_N + w₄·L_Surv
  • Poids dynamiques (wᵢ) ajustés automatiquement par Uncertainty Weighting
  • Les gradients de toutes les pertes remontent jusqu'au Bottleneck
  • Rétropropagation de L_Surv via la cross-attention → Bottleneck
  • Warm-up segmentation avant entraînement multitâche complet
═══════════════════════════════════════════════════════════════════════════════



```markdown
hecktor2026/
│
├── train.py                     # point d'entrée unique
│
├── src/
│   ├── dataset.py               # HECKTORDataset
│   ├── transforms.py            # augmentations MONAI
│   ├── preprocessing.py         # resampling, crop
│   │
│   ├── model.py                 # ← UN SEUL FICHIER central
│   │                            # MultitaskModel.forward() appelle tout
│   │                            # c'est LUI qui garantit la backprop end-to-end
│   │
│   ├── swinunetr.py             # SwinUNETRMultitask (sous-classe MONAI)
│   ├── heads.py                 # TNHead + SurvivalHead
│   ├── cross_attention.py       # CrossAttentionFusion
│   └── clinical_encoder.py     # MLP clinique
│
├── utils/
│   ├── losses.py                # DiceFocal + CrossEntropy + DeepHit
│   ├── uncertainty.py           # UncertaintyWeighting
│   └── metrics.py               # C-index, Dice
```

## Exemple de fichier model.py

```Python
class MultitaskModel(nn.Module):
    def __init__(self, config):
        self.backbone    = SwinUNETRMultitask(...)   # swinunetr.py
        self.t_head      = TNHead(...)               # heads.py
        self.n_head      = TNHead(...)               # heads.py
        self.clin_mlp    = ClinicalMLP(...)          # clinical_encoder.py
        self.proj_tn     = nn.Linear(8, d_model)
        self.cls_token   = nn.Parameter(torch.randn(1, 1, d_model))
        self.cross_attn  = CrossAttentionFusion(...) # cross_attention.py
        self.surv_head   = SurvivalHead(...)         # heads.py

    def forward(self, ct_pet, clinical):
        # 1. Backbone → masque + bottleneck
        seg_mask, bottleneck = self.backbone(ct_pet)

        # 2. T/N staging depuis le bottleneck
        t_logits = self.t_head(bottleneck)
        n_logits = self.n_head(bottleneck)

        # 3. Tokens pour la cross-attention
        token_tn   = self.proj_tn(
            torch.cat([t_logits, n_logits], dim=1)
        ).unsqueeze(1)
        token_clin = self.clin_mlp(clinical)
        cls        = self.cls_token.expand(ct_pet.size(0), -1, -1)
        Q          = torch.cat([cls, token_clin, token_tn], dim=1)

        # 4. Cross-attention
        cls_out = self.cross_attn(Q, bottleneck)

        # 5. Survie
        logits = self.surv_head(cls_out)

        return {
            "seg_mask":  seg_mask,
            "t_logits":  t_logits,
            "n_logits":  n_logits,
            "surv_logits": logits,
        }
```
## Exemple de train.py
```Python
model     = MultitaskModel(config)
weighting = UncertaintyWeighting(n_tasks=4)
optimizer = Adam(
    list(model.parameters()) + list(weighting.parameters())
)

for batch in dataloader:
    out = model(batch["ct_pet"], batch["clinical"])

    losses = [
        seg_loss(out["seg_mask"],    batch["seg_gt"]),
        ce_loss(out["t_logits"],     batch["t_label"]),
        ce_loss(out["n_logits"],     batch["n_label"]),
        deephit_loss(out["surv_logits"], batch["time"], batch["event"]),
    ]

    L_total = weighting(losses)  # poids dynamiques

    optimizer.zero_grad()
    L_total.backward()           # ← UN SEUL backward, remonte tout
    optimizer.step()
```