"""HECKTORDataset — images preprocessed (.npz) + cibles multitâche depuis le CSV."""

import os
import numpy as np
import pandas as pd
import torch
from typing import List, Optional, Dict
from monai.data import CacheDataset, DataLoader

from src.transforms import (
    get_multitask_train_transforms as get_train_transforms,
    get_multitask_validation_transforms as get_validation_transforms,
)


# Colonnes utilisées pour le bras clinique (7 features — cf. README)
CLINICAL_NUMERIC = ["Age"]
CLINICAL_CATEGORICAL = [
    "Gender",
    "Tobacco Consumption",
    "Alcohol Consumption",
    "Performance Status",
    "HPV Status",
    "M-stage",
]
assert len(CLINICAL_NUMERIC) + len(CLINICAL_CATEGORICAL) == 7

T_STAGES = ["T1", "T2", "T3", "T4"]
N_STAGES = ["N0", "N1", "N2", "N3"]


class ClinicalEncoder:
    """
    Encode 7 colonnes cliniques en un vecteur (B, 7) numérique.
      - continues : imputation médiane + standardisation
      - catégorielles : ordinal-encoding ; NaN → classe "Inconnu"
    """

    def __init__(self):
        self.age_median: Optional[float] = None
        self.age_mean: Optional[float] = None
        self.age_std: Optional[float] = None
        self.cat_maps: Dict[str, Dict[str, int]] = {}

    def fit(self, df: pd.DataFrame):
        ages = df["Age"].dropna().values.astype(float)
        self.age_median = float(np.median(ages)) if len(ages) else 0.0
        self.age_mean   = float(ages.mean()) if len(ages) else 0.0
        self.age_std    = float(ages.std()) if len(ages) and ages.std() > 1e-6 else 1.0

        for col in CLINICAL_CATEGORICAL:
            vals = df[col].astype(str).fillna("Inconnu").unique().tolist()
            if "Inconnu" not in vals:
                vals.append("Inconnu")
            self.cat_maps[col] = {v: i for i, v in enumerate(sorted(vals))}
        return self

    def transform_row(self, row: pd.Series) -> np.ndarray:
        # Age
        age = row.get("Age", np.nan)
        if pd.isna(age):
            age = self.age_median
        age_z = (float(age) - self.age_mean) / self.age_std

        feats = [age_z]
        for col in CLINICAL_CATEGORICAL:
            val = row.get(col, np.nan)
            if pd.isna(val):
                val = "Inconnu"
            val = str(val)
            mapping = self.cat_maps[col]
            idx = mapping.get(val, mapping["Inconnu"])
            feats.append(float(idx))
        return np.array(feats, dtype=np.float32)


def _encode_stage(value, stages: List[str], unknown_idx: int = 0) -> int:
    """Encode T/N en label int ; valeurs inconnues → unknown_idx (utilisé comme ignore_index)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return -1   # PyTorch CE ignore_index=-1
    s = str(value).strip().upper()
    # collapse N2a/N2b/N2c → N2
    if s.startswith("N2"):
        s = "N2"
    return stages.index(s) if s in stages else -1


class HECKTORMultitaskDataset(CacheDataset):
    """
    Wrappe CacheDataset MONAI pour les images, et injecte les cibles tabulaires
    (clinique, T, N, RFS, événement) dans chaque échantillon.
    """

    def __init__(self, data_list, transform, cache_rate, num_workers):
        super().__init__(data=data_list, transform=transform,
                         cache_rate=cache_rate, num_workers=num_workers)

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        # Quand RandCropByLabelClassesd renvoie plusieurs samples, MONAI fournit une liste.
        if isinstance(item, list):
            for s in item:
                s.update(self._tabular(s))
            return item
        item.update(self._tabular(item))
        return item

    @staticmethod
    def _tabular(sample):
        # Les champs scalaires/cliniques étaient stockés en numpy via le dict initial,
        # CacheDataset les a conservés tels quels (la liste de transforms les ignore).
        return {}


def _build_data_list(case_ids, images_dir, labels_dir, df, clinical_encoder) -> List[dict]:
    items = []
    df_idx = df.set_index("PatientID")
    for cid in case_ids:
        if cid not in df_idx.index:
            continue
        row = df_idx.loc[cid]
        clin = clinical_encoder.transform_row(row)
        t_lbl = _encode_stage(row.get("T_stage"), T_STAGES)
        n_lbl = _encode_stage(row.get("N_stage"), N_STAGES)
        rfs   = float(row.get("RFS", np.nan)) if not pd.isna(row.get("RFS", np.nan)) else 0.0
        evt   = int(row.get("Relapse", 0)) if not pd.isna(row.get("Relapse", np.nan)) else 0
        items.append({
            "ct":       os.path.join(images_dir, f"{cid}_ct.npz"),
            "pet":      os.path.join(images_dir, f"{cid}_pet.npz"),
            "label":    os.path.join(labels_dir, f"{cid}_label.npz"),
            "clinical": torch.from_numpy(clin),
            "t_label":  torch.tensor(t_lbl, dtype=torch.long),
            "n_label":  torch.tensor(n_lbl, dtype=torch.long),
            "time":     torch.tensor(rfs, dtype=torch.float32),
            "event":    torch.tensor(evt, dtype=torch.float32),
            "case_id":  cid,
        })
    return items


def get_multitask_dataloaders(config) -> tuple:
    """
    Splits train/val + DataLoaders MONAI. Renvoie (train_loader, val_loader, train_df).
    train_df sert ensuite à calculer les quantiles de discrétisation du temps.
    """
    import random

    images_dir = os.path.join(config.data_root, config.train_images_dir)
    labels_dir = os.path.join(config.data_root, config.train_labels_dir)
    ct_files = sorted(f for f in os.listdir(images_dir) if f.endswith("_ct.npz"))
    case_ids = [f.replace("_ct.npz", "") for f in ct_files]

    random.seed(config.seed)
    random.shuffle(case_ids)
    n_val = int(len(case_ids) * config.val_split)
    val_ids   = case_ids[:n_val]
    train_ids = case_ids[n_val:]
    print(f"[Data] {len(train_ids)} train / {len(val_ids)} val")

    df = pd.read_csv(config.csv_path)

    # Encodeur clinique fit sur le train uniquement
    clin_enc = ClinicalEncoder().fit(df[df["PatientID"].isin(train_ids)])

    train_items = _build_data_list(train_ids, images_dir, labels_dir, df, clin_enc)
    val_items   = _build_data_list(val_ids,   images_dir, labels_dir, df, clin_enc)

    # df d'entraînement (pour quantiles temps)
    train_df = df[df["PatientID"].isin([it["case_id"] for it in train_items])].copy()

    train_ds = HECKTORMultitaskDataset(
        data_list=train_items,
        transform=get_train_transforms(config),
        cache_rate=config.cache_rate,
        num_workers=config.num_workers,
    )
    val_ds = HECKTORMultitaskDataset(
        data_list=val_items,
        transform=get_validation_transforms(),
        cache_rate=config.cache_rate,
        num_workers=config.num_workers,
    )

    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=config.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=True,
        persistent_workers=config.num_workers > 0,
    )
    return train_loader, val_loader, train_df, clin_enc
