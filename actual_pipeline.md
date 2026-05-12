# HECKTOR 2026 

### Aproches précédentes
- Les trois tâches du challenge (segmentation, staging, pronostic) sont **complètement isolées** : aucune information ne circule entre elles.
- L'image est dégradée à 96³, sans information sur la localisation tumorale.
- Pas d'utilisation des masques de segmentation pour le pronostic.
---

### Pipeline précédente pour la prédiction de la survie

```
CT+PET (2 ch, 96³)              Clinical (clin_dim)
        │                                │
        ▼                                ▼
   ResNet-18 3D                    MLP clinique
   (FC → Identity)              (64→32, BN, Dropout)
        │                                │
        ▼                                ▼
   f_img (512)                      f_clin (32)
        │                                │
        └──────────► Concat ◄────────────┘
                       │
                  544 dims
                       │
                       ▼
                  MLP fusion
              (544→512→256→128)
                       │
                       ▼
                features (128)
                  │        │
                  ▼        ▼
            Risk head   ┌─ BaggedIcareSurvival
            (128→1)     │  (icare, post-hoc,
                  │     │   non différentiable)
                  ▼     │
        DeepHit+Contrast│
              loss      ▼
                  risk_score final
```


Participants are invited to develop a multimodal pipeline leveraging FDG PET, CT, and clinical data to:

    Segment primary tumors and lymph nodes
    Infer radiological TN staging
    Predict recurrence-free survival

This unified task reflects a realistic clinical workflow, integrating diagnosis, staging, and prognosis into a single framework.

Multi-Task Learning
Pretrained SegMamba model + LoRA



Backbones Image : Modèles Mamba 3D (pré-entraînés sur des centaines de milliers de scanners) congelés + LoRA.

Backbone Clinique : Un Tabular Transformer (pour traiter l'âge, le sexe, le stade comme des tokens textuels) plutôt qu'un vieux MLP.

Tâches Auxiliaires : Le réseau est forcé de prédire le masque de segmentation et le statut HPV/Stade T pour s'assurer qu'il "regarde" la bonne chose.

Module de Fusion : Un bloc de Cross-Attention qui fait dialoguer les tokens images et cliniques.

Tête de Survie : Un modèle en temps discret (comme DeepHit) plutôt qu'un modèle de Cox continu, pour mieux modéliser les risques non proportionnels.

End-to-End Multitask
Warm-up
rétropropagation du gradient de survie à travers les tâches intermédiaires


### Pipeline 2026 pour la prédiction de la survie

CT+PET (2 canaux, 128³ ou 256³)
        │
        ▼
┌─────────────────────────────────┐
│SegMamba Encoder Pré-entraîné SSL│ 
│      (Poids gelés + LoRA)       │ 
└─────────────────┬───────────────┘
                  │
                  ▼
         Bottleneck (Latent z)
                  │
  ┌───────────────┼───────────────┐
  │               │               │
  ▼               ▼               ▼
Décodeur       T-Head          N-Head         
(Mamba)        (Linear)        (Linear)       
  │               │               │               
  ▼               ▼               ▼               
Masque          Logits T        Logits N      
  │               │               │               
  │ L_Seg         │ L_T           │ L_N           
  │               │               │               
  │               └───────┬───────┘                      
  │                       │                              ┌─────────────────────┐
  │                       │                              │ Données Cliniques   │
  │                       │                              │ (Âge, Sexe, Poids,  │
  │                       │                              │  **Statut HPV**)    │
  │                       │                              └─────────┬───────────┘
  │                       │                                        │
  |                       |                              Tabular Transformer
  │                       ▼                                        ▼
  │               ┌────────────────────────────────────────────────────────┐
  └──────────────►│                Cross-Attention Fusion                  │
                  │  Queries : Tokens Cliniques (contenant l'HPV), T, N    │
                  │  Keys/Values : Bottleneck z + Masque (L'image)         │
                  └───────────────────────┬────────────────────────────────┘
                                          │
                                          ▼
                            Tête de Survie (Discrete-Time)
                                          │
                                          ▼
                             Risk Probabilities ──► L_Surv (DeepHit)

========================================================================================
Fonction de perte totale (End-to-End) :
L_Total = w₁*L_Seg + w₂*L_T + w₃*L_N + w₄*L_Surv

* utilisation d'un Warm-up pour la segmentation
* Les gradients de toutes les pertes remontent jusqu'au Bottleneck.
* Les poids dynamiques (w) sont ajustés automatiquement par incertitude (Uncertainty Weighting).