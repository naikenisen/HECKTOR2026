import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.metrics import balanced_accuracy_score, recall_score
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, ToTensord, Resized
)
from monai.networks.nets import resnet18

# =============================================================================
# Configuration
# =============================================================================

PATH_TO_TRAINING_IMAGES = "../../hecktor2026_training"
PATH_TO_CSV = "../../hecktor2026_training/HECKTOR_2026_Training.csv"

# TN stages per AJCC/UICC 7th Edition (N2b and N2c collapsed to N2)
T_STAGES = ["T1", "T2", "T3", "T4"]
N_STAGES = ["N0", "N1", "N2", "N3"]

# =============================================================================
# Dataset
# =============================================================================

class HecktorTNDataset(Dataset):
    def __init__(self, csv_file, img_dir, transforms, patient_ids=None,
                 scaler=None, ohe=None, t_encoder=None, n_encoder=None):
        self.df_full = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transforms = transforms

        if patient_ids is not None:
            self.df = self.df_full[self.df_full["PatientID"].isin(patient_ids)].reset_index(drop=True)
        else:
            self.df = self.df_full.copy()

        self.patient_ids = self.df["PatientID"].tolist()

        # Encode T and N stage labels
        self.t_encoder = t_encoder or LabelEncoder().fit(T_STAGES)
        self.n_encoder = n_encoder or LabelEncoder().fit(N_STAGES)
        self.t_labels = self.t_encoder.transform(self.df["T_stage"].astype(str).values)
        self.n_labels = self.n_encoder.transform(self.df["N_stage"].astype(str).values)

        # Clinical features
        num_cols = ["Age"]
        cat_cols = ["Gender", "Tobacco Consumption", "Alcohol Consumption",
                    "Performance Status", "HPV Status"]

        num_data = self.df[num_cols].fillna(0).values
        self.scaler = scaler or StandardScaler()
        self.num = self.scaler.fit_transform(num_data) if scaler is None else self.scaler.transform(num_data)

        df_cat = self.df[cat_cols].astype(str).fillna("Unknown")
        self.ohe = ohe or OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.cat = self.ohe.fit_transform(df_cat) if ohe is None else self.ohe.transform(df_cat)

        self.clinical_feats = np.hstack([self.num, self.cat])

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        pid = self.patient_ids[idx]
        ct_path = os.path.join(self.img_dir, pid, f"{pid}__CT.nii.gz")
        pet_path = os.path.join(self.img_dir, pid, f"{pid}__PT.nii.gz")

        data = self.transforms({"ct": ct_path, "pet": pet_path})
        x_img = torch.cat([data["ct"], data["pet"]], dim=0)
        x_clin = torch.tensor(self.clinical_feats[idx], dtype=torch.float32)
        t_label = torch.tensor(self.t_labels[idx], dtype=torch.long)
        n_label = torch.tensor(self.n_labels[idx], dtype=torch.long)
        return x_img, x_clin, t_label, n_label


# =============================================================================
# Image transforms
# =============================================================================

img_transforms = Compose([
    LoadImaged(keys=["ct", "pet"]),
    EnsureChannelFirstd(keys=["ct", "pet"]),
    ScaleIntensityd(keys=["ct", "pet"]),
    Resized(keys=["ct", "pet"], spatial_size=(96, 96, 96)),
    ToTensord(keys=["ct", "pet"]),
])


# =============================================================================
# Model: dual-head multimodal ResNet
# =============================================================================

class MultiModalTNModel(nn.Module):
    """
    Multimodal ResNet18 backbone with two classification heads:
    one for T stage and one for N stage.
    """
    def __init__(self, clin_feat_dim, num_t_classes=4, num_n_classes=4):
        super().__init__()
        self.img_model = resnet18(
            spatial_dims=3,
            n_input_channels=2,
            num_classes=512,
        )
        self.img_model.fc = nn.Identity()

        self.clin_model = nn.Sequential(
            nn.Linear(clin_feat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        fused_dim = 512 + 32
        self.t_head = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_t_classes),
        )
        self.n_head = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_n_classes),
        )

    def forward(self, x_img, x_clin):
        f_img = self.img_model(x_img)       # [B, 512]
        f_clin = self.clin_model(x_clin)    # [B, 32]
        f = torch.cat([f_img, f_clin], dim=1)
        return self.t_head(f), self.n_head(f)


# =============================================================================
# Training
# =============================================================================

def train(csv_path, img_dir, num_epochs=10, batch_size=4, val_split=0.15, test_split=0.15):
    os.makedirs("tn_logs", exist_ok=True)

    df_all = pd.read_csv(csv_path)
    all_pids = df_all["PatientID"].values
    all_strata = df_all["T_stage"].astype(str) + "_" + df_all["N_stage"].astype(str)

    # Fit global preprocessors on all data
    num_cols = ["Age"]
    cat_cols = ["Gender", "Tobacco Consumption", "Alcohol Consumption",
                "Performance Status", "HPV Status"]
    scaler = StandardScaler().fit(df_all[num_cols].fillna(0).values)
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(
        df_all[cat_cols].astype(str).fillna("Unknown").values
    )
    t_encoder = LabelEncoder().fit(T_STAGES)
    n_encoder = LabelEncoder().fit(N_STAGES)

    # Three-way split: train / val / test
    pids_trainval, pids_test, strat_trainval, _ = train_test_split(
        all_pids, all_strata, test_size=test_split, stratify=all_strata, random_state=42
    )
    val_fraction = val_split / (1.0 - test_split)
    pids_train, pids_val = train_test_split(
        pids_trainval, test_size=val_fraction, stratify=strat_trainval, random_state=42
    )
    print(f"Split — train: {len(pids_train)}, val: {len(pids_val)}, test: {len(pids_test)}")

    train_ds = HecktorTNDataset(csv_path, img_dir, img_transforms,
                                patient_ids=pids_train, scaler=scaler, ohe=ohe,
                                t_encoder=t_encoder, n_encoder=n_encoder)
    val_ds   = HecktorTNDataset(csv_path, img_dir, img_transforms,
                                patient_ids=pids_val, scaler=scaler, ohe=ohe,
                                t_encoder=t_encoder, n_encoder=n_encoder)
    test_ds  = HecktorTNDataset(csv_path, img_dir, img_transforms,
                                patient_ids=pids_test, scaler=scaler, ohe=ohe,
                                t_encoder=t_encoder, n_encoder=n_encoder)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiModalTNModel(
        clin_feat_dim=train_ds.clinical_feats.shape[1],
        num_t_classes=len(T_STAGES),
        num_n_classes=len(N_STAGES),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    best_val_score = 0.0
    best_model_path = "tn_logs/best_model.pt"

    for epoch in range(1, num_epochs + 1):
        model.train()
        for x_img, x_clin, t_lbl, n_lbl in train_loader:
            x_img, x_clin = x_img.to(device), x_clin.to(device)
            t_lbl, n_lbl  = t_lbl.to(device),  n_lbl.to(device)
            optimizer.zero_grad()
            t_logits, n_logits = model(x_img, x_clin)
            loss = criterion(t_logits, t_lbl) + criterion(n_logits, n_lbl)
            loss.backward()
            optimizer.step()

        model.eval()
        all_t_true, all_t_pred, all_n_true, all_n_pred = [], [], [], []
        with torch.no_grad():
            for x_img, x_clin, t_lbl, n_lbl in val_loader:
                x_img, x_clin = x_img.to(device), x_clin.to(device)
                t_logits, n_logits = model(x_img, x_clin)
                all_t_true.extend(t_lbl.numpy())
                all_t_pred.extend(t_logits.cpu().argmax(dim=1).numpy())
                all_n_true.extend(n_lbl.numpy())
                all_n_pred.extend(n_logits.cpu().argmax(dim=1).numpy())

        bal_acc_t = balanced_accuracy_score(all_t_true, all_t_pred)
        bal_acc_n = balanced_accuracy_score(all_n_true, all_n_pred)
        mean_bal_acc = (bal_acc_t + bal_acc_n) / 2
        print(f"Epoch {epoch:02d} | Val BalAcc T: {bal_acc_t:.4f}  N: {bal_acc_n:.4f}  Mean: {mean_bal_acc:.4f}")

        if mean_bal_acc > best_val_score:
            best_val_score = mean_bal_acc
            torch.save(model.state_dict(), best_model_path)

    # Final evaluation on test set
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    all_t_true, all_t_pred, all_n_true, all_n_pred = [], [], [], []
    with torch.no_grad():
        for x_img, x_clin, t_lbl, n_lbl in test_loader:
            x_img, x_clin = x_img.to(device), x_clin.to(device)
            t_logits, n_logits = model(x_img, x_clin)
            all_t_true.extend(t_lbl.numpy())
            all_t_pred.extend(t_logits.cpu().argmax(dim=1).numpy())
            all_n_true.extend(n_lbl.numpy())
            all_n_pred.extend(n_logits.cpu().argmax(dim=1).numpy())

    print(f"\nTest BalAcc  T: {balanced_accuracy_score(all_t_true, all_t_pred):.4f}"
          f"  N: {balanced_accuracy_score(all_n_true, all_n_pred):.4f}")
    print(f"Test Recall  T: {recall_score(all_t_true, all_t_pred, average='macro', zero_division=0):.4f}"
          f"  N: {recall_score(all_n_true, all_n_pred, average='macro', zero_division=0):.4f}")


if __name__ == "__main__":
    train(
        csv_path=PATH_TO_CSV,
        img_dir=PATH_TO_TRAINING_IMAGES,
        num_epochs=10,
        batch_size=4,
    )
