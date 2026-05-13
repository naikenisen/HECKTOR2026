# HECKTOR 2026 

## Previous approaches
- The three challenge tasks (segmentation, staging, prognosis) are **completely isolated**: no information flows between them.
- The image is downgraded to 96³, without any information about tumor localisation.

## 2026 Guidelines
Participants are invited to develop a multimodal pipeline leveraging FDG PET, CT, and clinical data to:

    Segment primary tumors and lymph nodes
    Infer radiological TN staging
    Predict recurrence-free survival

This unified task reflects a realistic clinical workflow, integrating diagnosis, staging, and prognosis into a single framework.

---

## 2026 Pipeline — End-to-End Multitask Learning

```
  CT+PET (RAW)
        |
        | preprocessing (src/preprocessing.py)
        | resampling 2×2×2 mm³, crop 128³ centred on tumour
        |
CT+PET (B, 2, 128, 128, 128)
        |
        | dataloading and transformation (src/dataloader.py, src/transforms.py)
        | missing values: categorical variables → "Unknown" class
        | continuous variables → normalisation + median imputation
        │
        ▼
    SwinUNETR (src/swinunetr.py)
    SSL pre-trained weights on 5050 CTs (model_swinvit.pt)
        │
    L_Seg = Dice+Focal (utils/losses.py)
        │
        ├──► seg_mask (B, 3, D, H, W)
        │
        └──► bottleneck (B, C, D', H', W')
                  │
        ┌─────────┼─────────┐
        │                   │
        ▼                   ▼
T-Head (src/heads.py)    N-Head (src/heads.py)
  GAP → hidden (B,256)    GAP → hidden (B,256)
      → logits (B,4)          → logits (B,4)
        │                   │
   L_T = CrossEnt      L_N = CrossEnt  (utils/losses.py)
        │                   │
        └─────────┬─────────┘
                  │
                  │  t_feat (B,256) + n_feat (B,256)
                  │  → nn.Linear(512, d_model)
                  │  → token_tn (B, 1, d_model)
                  │
                  │          Clinical (B, 7)
                  │               │
                  │     MLP (7→64→d_model) (src/clinical_encoder.py)
                  │     NaN → "Unknown" class for categoricals
                  │     NaN → median imputation for continuous
                  │               │
                  │          token_clin (B, 1, d_model)
                  │               │
                  ▼               ▼
      ┌─────────────────────────────────────────────────┐
      │              Cross-Attention Fusion              │
      │              (src/cross_attention.py)            │
      │                                                  │
      │  Q (B, 3, d_model) :                             │
      │  cat([CLS, token_clin, token_tn], dim=1)         │
      │                                                  │
      │  K = V (B, N, d_model) :                         │
      │  Linear(C, d_model)(bottleneck.flatten(2)        │
      │  .permute(0,2,1))                                │
      │                                                  │
      │  attn(Q, K, V) → enriched CLS (B, d_model)       │
      └─────────────────────────┬───────────────────────┘
                                │
                                ▼
                Survival Head (Discrete-Time)
                nn.Linear(d_model → 256 → T) (src/heads.py)
                T intervals defined by quantiles
                over event times from the train set
                                │
                    ┌───────────┴────────────┐
                    │                        │
                    ▼ (training)             ▼ (inference)
             raw logits (B, T)          softmax(logits)
             L_Surv = DeepHit               │
             (utils/losses.py)         Risk Probabilities (B, T)
             gradient clipping
             max_norm = 1.0

═══════════════════════════════════════════════════════════════════════════════

TOTAL LOSS FUNCTION (End-to-End):

  L_Total = w₁·L_Seg + w₂·L_T + w₃·L_N + w₄·L_Surv

  • Dynamic weights (wᵢ) adjusted by Uncertainty Weighting (Kendall et al.)
  • Gradients from all losses backpropagate through the Bottleneck
  • Backpropagation of L_Surv via cross-attention → Bottleneck
  • Warm-up: T/N and Survival heads are frozen for N_warmup epochs
    → only segmentation is trained
    → then progressive unfreezing of all heads

═══════════════════════════════════════════════════════════════════════════════
```

---

## Repository structure

```
hecktor2026/
│
├── train.py                     # single entry point
│
├── src/
│   ├── dataset.py               # HECKTORDataset + missing value handling
│   ├── transforms.py            # MONAI augmentations (flip, noise, intensity)
│   ├── preprocessing.py         # resampling 2×2×2 mm³, crop 128³
│   │
│   ├── model.py                 # MultitaskModel — central file
│   │                            # forward() connects all components
│   │                            # guarantees end-to-end backprop
│   │
│   ├── swinunetr.py             # SwinUNETRMultitask (MONAI subclass)
│   │                            # returns (seg_mask, bottleneck)
│   ├── heads.py                 # TNHead (returns feat + logits) + SurvivalHead
│   ├── cross_attention.py       # CrossAttentionFusion
│   └── clinical_encoder.py     # Clinical MLP (7 → 64 → d_model)
│
├── utils/
│   ├── losses.py                # DiceFocal + CrossEntropy + DeepHit
│   └── metrics.py               # C-index, Dice, Balanced Accuracy
│
└── config.py                    # d_model, T, N_warmup, lr, batch_size
```

---

## src/model.py

```python
class MultitaskModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.backbone   = SwinUNETRMultitask(...)    # swinunetr.py
        self.t_head     = TNHead(...)                # heads.py — returns (feat, logits)
        self.n_head     = TNHead(...)                # heads.py — returns (feat, logits)
        self.clin_mlp   = ClinicalMLP(...)           # clinical_encoder.py
        self.proj_tn    = nn.Linear(512, d_model)    # 256+256 → d_model
        self.cls_token  = nn.Parameter(torch.randn(1, 1, d_model))
        self.cross_attn = CrossAttentionFusion(...)  # cross_attention.py
        self.surv_head  = SurvivalHead(...)          # heads.py

    def forward(self, ct_pet, clinical):
        # 1. Backbone → mask + bottleneck
        seg_mask, bottleneck = self.backbone(ct_pet)

        # 2. T/N staging — intermediate features + logits
        t_feat, t_logits = self.t_head(bottleneck)   # (B,256), (B,4)
        n_feat, n_logits = self.n_head(bottleneck)   # (B,256), (B,4)

        # 3. TN token from rich features (not logits)
        token_tn = self.proj_tn(
            torch.cat([t_feat, n_feat], dim=1)       # (B, 512)
        ).unsqueeze(1)                               # (B, 1, d_model)

        # 4. Clinical token
        token_clin = self.clin_mlp(clinical)         # (B, 1, d_model)

        # 5. CLS token
        cls = self.cls_token.expand(ct_pet.size(0), -1, -1)

        # 6. Queries
        Q = torch.cat([cls, token_clin, token_tn], dim=1)  # (B, 3, d_model)

        # 7. Cross-attention — K=V from flattened + projected bottleneck
        cls_out = self.cross_attn(Q, bottleneck)     # (B, d_model)

        # 8. Survival → raw logits (softmax applied only at inference)
        surv_logits = self.surv_head(cls_out)        # (B, T)

        return {
            "seg_mask":    seg_mask,
            "t_logits":    t_logits,
            "n_logits":    n_logits,
            "surv_logits": surv_logits,
        }
```

---

## train.py

```python
# Discretise time intervals by quantiles
event_times = train_df.loc[train_df["event"]==1, "time"].values
cuts = np.quantile(event_times, np.linspace(0, 1, T+1))  # T equiprobable intervals

model     = MultitaskModel(config)
weighting = UncertaintyWeighting(n_tasks=4)
optimizer = Adam(list(model.parameters()) + list(weighting.parameters()))

for epoch in range(total_epochs):

    # Phase 1 — Warm-up: freeze T/N and Survival heads
    if epoch < config.N_warmup:
        for p in model.t_head.parameters():     p.requires_grad = False
        for p in model.n_head.parameters():     p.requires_grad = False
        for p in model.surv_head.parameters():  p.requires_grad = False
        for p in model.cross_attn.parameters(): p.requires_grad = False
    else:
        for p in model.parameters(): p.requires_grad = True

    for batch in dataloader:
        out = model(batch["ct_pet"], batch["clinical"])

        losses = [
            seg_loss(out["seg_mask"],      batch["seg_gt"]),
            ce_loss(out["t_logits"],       batch["t_label"]),
            ce_loss(out["n_logits"],       batch["n_label"]),
            deephit_loss(out["surv_logits"], batch["time"], batch["event"]),
        ]

        L_total = weighting(losses)

        optimizer.zero_grad()
        L_total.backward()

        # Gradient clipping on DeepHit to prevent explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
```
