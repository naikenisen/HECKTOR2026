import os
import sys
import pandas as pd
import numpy as np
import torch
import pickle
import random
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from lifelines.utils import concordance_index
from tqdm import tqdm
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, ToTensord, Resized
)
from monai.networks.nets import resnet18

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "utils"))
from losses import DeepHitLoss

# =============================================================================
# Configuration and Constants
# =============================================================================

RANDOM_SEED = 42
IMAGE_SIZE = (96, 96, 96)
KNN_NEIGHBORS = 5
CONCORDANCE_EPSILON = 1e-8

# Data paths - update these for your environment
TRAINING_IMAGES_PATH = "../../hecktor2026_training"
EHR_DATA_PATH = "../../hecktor2026_training/HECKTOR_2026_Training.csv"


# =============================================================================
# Utility Functions
# =============================================================================

def set_random_seed(seed=RANDOM_SEED):
    """Set random seeds for reproducible results across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_device():
    """Configure and return the appropriate device for computation."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    return device

# =============================================================================
# Loss Functions for Feature Extractor Training
# =============================================================================

class SurvivalContrastiveLoss(nn.Module):
    """
    Contrastive loss for survival analysis that encourages similar survival times
    to have similar representations and different survival times to be separated.
    """
    def __init__(self, margin=1.0, temperature=0.1):
        super().__init__()
        self.margin = margin
        self.temperature = temperature

    def forward(self, features, survival_times, event_indicators):
        """
        Args:
            features: Extracted features [batch_size, feature_dim]
            survival_times: Survival times [batch_size]
            event_indicators: Event indicators [batch_size]
        """
        batch_size = features.size(0)
        
        # Normalize features
        features = nn.functional.normalize(features, p=2, dim=1)
        
        # Compute pairwise distances
        distance_matrix = torch.cdist(features, features, p=2)
        
        # Create similarity targets based on survival times
        time_diff_matrix = torch.abs(survival_times.unsqueeze(0) - survival_times.unsqueeze(1))
        
        # Similar pairs: small time differences and both have events
        event_matrix = event_indicators.unsqueeze(0) * event_indicators.unsqueeze(1)
        similar_mask = (time_diff_matrix < torch.median(time_diff_matrix)) & (event_matrix > 0)
        
        # Dissimilar pairs: large time differences
        dissimilar_mask = time_diff_matrix > torch.quantile(time_diff_matrix, 0.75)
        
        # Remove diagonal
        eye_mask = torch.eye(batch_size, device=features.device).bool()
        similar_mask = similar_mask & ~eye_mask
        dissimilar_mask = dissimilar_mask & ~eye_mask
        
        loss = torch.tensor(0.0, device=features.device, requires_grad=True)
        
        if similar_mask.sum() > 0:
            # Similar pairs should be close
            similar_distances = distance_matrix[similar_mask]
            similar_loss = similar_distances.mean()
            loss = loss + similar_loss
        
        if dissimilar_mask.sum() > 0:
            # Dissimilar pairs should be far apart
            dissimilar_distances = distance_matrix[dissimilar_mask]
            dissimilar_loss = torch.clamp(self.margin - dissimilar_distances, min=0).mean()
            loss = loss + dissimilar_loss
        
        return loss

# =============================================================================
# Dataset Class
# =============================================================================

class HecktorSurvivalDataset(Dataset):
    """Dataset class for pre-loaded HECKTOR survival data."""
    def __init__(self, cached_data, patient_ids):
        self.patient_ids = [pid for pid in patient_ids if pid in cached_data['images']]
        self.images = cached_data['images']
        
        self.clinical_features = cached_data['clinical_features']['features']
            
        self.survival_data = cached_data['survival_data']
        
        print(f"Dataset initialized with {len(self.patient_ids)} patients")

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        
        image_tensor = self.images[patient_id]
        clinical_tensor = torch.tensor(self.clinical_features[patient_id], dtype=torch.float32)
        survival_time = torch.tensor(self.survival_data[patient_id]['time'], dtype=torch.float32)
        event_indicator = torch.tensor(self.survival_data[patient_id]['event'], dtype=torch.float32)
        
        return image_tensor, clinical_tensor, survival_time, event_indicator

# =============================================================================
# BaggedIcareSurvival Model
# =============================================================================

class FusedFeatureExtractor(nn.Module):
    """
    Feature extractor specifically designed for BaggedIcareSurvival.
    Combines 3D medical imaging and clinical data into rich survival features.
    """
    def __init__(self, clinical_feature_dim, feature_output_dim=128):
        super().__init__()
        
        # Store dimensions for saving
        self.clinical_feature_dim = clinical_feature_dim
        self.feature_output_dim = feature_output_dim
        
        # 3D ResNet-18 for combined CT+PET input
        self.imaging_backbone = resnet18(
            spatial_dims=3,
            n_input_channels=2,
            num_classes=1,
        )
        self.imaging_backbone.fc = nn.Identity()

        # Clinical data processor with deeper architecture
        self.clinical_processor = nn.Sequential(
            nn.Linear(clinical_feature_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        # Feature fusion with multiple pathways
        self.feature_fusion = nn.Sequential(
            nn.Linear(512 + 32, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, feature_output_dim)
        )
        
        # Risk prediction head for training guidance
        self.risk_head = nn.Sequential(
            nn.Linear(feature_output_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, medical_images, clinical_features, return_risk=False):
        # Extract imaging features
        imaging_features = self.imaging_backbone(medical_images)
        
        # Process clinical features
        clinical_features_processed = self.clinical_processor(clinical_features)
        
        # Combine and fuse
        combined_features = torch.cat([imaging_features, clinical_features_processed], dim=1)
        fused_features = self.feature_fusion(combined_features)
        
        if return_risk:
            risk_scores = self.risk_head(fused_features).squeeze(-1)
            return fused_features, risk_scores
        
        return fused_features

class HecktorSurvivalModel:
    """Trains FusedFeatureExtractor end-to-end with DeepHit + contrastive loss."""

    def __init__(self, clinical_feature_dim, device, feature_dim=128):
        self.device = device
        self.feature_dim = feature_dim
        self.clinical_feature_dim = clinical_feature_dim

        self.feature_extractor = FusedFeatureExtractor(
            clinical_feature_dim, feature_dim
        ).to(device)

        self.optimizer = optim.Adam(
            self.feature_extractor.parameters(), lr=1e-3, weight_decay=1e-5
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5
        )

        self.survival_loss   = DeepHitLoss(ranking_weight=0.3)
        self.contrastive_loss = SurvivalContrastiveLoss(margin=2.0, temperature=0.1)

        self.best_c_index      = 0.0
        self.best_model_state  = None

    def _train_epoch(self, train_loader):
        self.feature_extractor.train()
        total_loss, batch_count = 0.0, 0

        for images, clinical, times, events in tqdm(train_loader, desc="  batches", leave=False):
            images, clinical = images.to(self.device), clinical.to(self.device)
            times,  events   = times.to(self.device),  events.to(self.device)

            self.optimizer.zero_grad()
            features, risk_scores = self.feature_extractor(images, clinical, return_risk=True)

            loss = (self.survival_loss(risk_scores, times, events)
                    + 0.1 * self.contrastive_loss(features, times, events))

            if not torch.isnan(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.feature_extractor.parameters(), 1.0)
                self.optimizer.step()
                total_loss += loss.item()
                batch_count += 1

        return total_loss / max(batch_count, 1)

    def evaluate(self, data_loader):
        """Return C-index from risk_head predictions."""
        self.feature_extractor.eval()
        all_risks, all_times, all_events = [], [], []

        with torch.no_grad():
            for images, clinical, times, events in data_loader:
                images, clinical = images.to(self.device), clinical.to(self.device)
                _, risk_scores = self.feature_extractor(images, clinical, return_risk=True)
                all_risks.extend(risk_scores.cpu().numpy())
                all_times.extend(times.numpy())
                all_events.extend(events.numpy())

        risks  = np.array(all_risks)
        times  = np.array(all_times)
        events = np.array(all_events)

        if events.sum() == 0:
            return 0.5
        return concordance_index(times, -risks, events)

    def fit(self, train_loader, val_loader, num_epochs=100):
        print(f"Training for {num_epochs} epochs")

        for epoch in tqdm(range(num_epochs), desc="Epochs"):
            avg_loss = self._train_epoch(train_loader)
            c_index  = self.evaluate(val_loader)
            self.scheduler.step(c_index)

            print(f"  Epoch {epoch + 1:3d} | loss {avg_loss:.4f} | val C-index {c_index:.4f}")

            if c_index > self.best_c_index:
                self.best_c_index     = c_index
                self.best_model_state = {k: v.clone() for k, v in
                                         self.feature_extractor.state_dict().items()}

        print(f"\nTraining complete. Best val C-index: {self.best_c_index:.4f}")
        if self.best_model_state is not None:
            self.feature_extractor.load_state_dict(self.best_model_state)

    def predict(self, data_loader):
        """Return (risk_scores, times, events) arrays."""
        self.feature_extractor.eval()
        all_risks, all_times, all_events = [], [], []

        with torch.no_grad():
            for images, clinical, times, events in data_loader:
                images, clinical = images.to(self.device), clinical.to(self.device)
                _, risk_scores = self.feature_extractor(images, clinical, return_risk=True)
                all_risks.extend(risk_scores.cpu().numpy())
                all_times.extend(times.numpy())
                all_events.extend(events.numpy())

        return np.array(all_risks), np.array(all_times), np.array(all_events)

    def save_model(self, filepath_prefix):
        torch.save(self.feature_extractor.state_dict(), f"{filepath_prefix}_feature_extractor.pt")
        print(f"Model saved: {filepath_prefix}_feature_extractor.pt")

    def load_model(self, filepath_prefix):
        self.feature_extractor.load_state_dict(
            torch.load(f"{filepath_prefix}_feature_extractor.pt", map_location=self.device)
        )
        print(f"Model loaded: {filepath_prefix}_feature_extractor.pt")

# =============================================================================
# Data Loading Functions (Updated)
# =============================================================================

def create_image_transforms():
    """Create MONAI transforms for CT and PET image preprocessing."""
    transforms = Compose([
        LoadImaged(keys=["ct","pet"]),
        EnsureChannelFirstd(keys=["ct", "pet"]),
        ScaleIntensityd(keys=["ct","pet"]),
        Resized(keys=["ct", "pet"], spatial_size=IMAGE_SIZE), 
        ToTensord(keys=["ct","pet"]),
    ])
    return transforms

def find_image_path(patient_id, modality, directories):
    """Find the file path for a specific patient and imaging modality."""
    for directory in directories:
        filename = f"{patient_id}__{modality}.nii.gz"
        full_path = os.path.join(directory, filename)
        if os.path.exists(full_path):
            return full_path
    raise FileNotFoundError(f"{modality} image for patient {patient_id} not found")


def preprocess_clinical_data(dataframe):
    """Preprocess clinical features with one-hot encoding and proper NaN handling."""
    
    # All clinical features used in preprocessing
    ALL_CLINICAL_FEATURES = [
        "Age", "Gender", "Tobacco Consumption", "Alcohol Consumption", 
        "Performance Status", "M-stage", "Treatment"
    ]
    
    # Categorical features for one-hot encoding (all except Age)
    CATEGORICAL_FEATURES = [
        "Gender", "Tobacco Consumption", "Alcohol Consumption", 
        "Performance Status", "M-stage", "Treatment"
    ]
    
    feature_subset = dataframe[ALL_CLINICAL_FEATURES].copy()
    
    # Handle Age (continuous variable)
    # Fill NaN values with median
    age_median = feature_subset["Age"].median()
    feature_subset["Age"] = feature_subset["Age"].fillna(age_median)
    
    # Standardize Age
    age_scaler = StandardScaler()
    age_scaled = age_scaler.fit_transform(feature_subset[["Age"]])
    
    # Handle categorical features with one-hot encoding
    # Fill NaN values with 'Unknown' category for each categorical feature
    categorical_data = feature_subset[CATEGORICAL_FEATURES].copy()
    for col in CATEGORICAL_FEATURES:
        categorical_data[col] = categorical_data[col].fillna('Unknown')
        # Convert to string to ensure consistent data type
        categorical_data[col] = categorical_data[col].astype(str)
    
    # Apply one-hot encoding
    # Use drop_first=False to keep all categories (including Unknown)
    categorical_encoded = pd.get_dummies(
        categorical_data, 
        columns=CATEGORICAL_FEATURES,
        prefix=CATEGORICAL_FEATURES,
        dummy_na=False,  # We already handled NaN by filling with 'Unknown'
        drop_first=False  # Keep all categories for completeness
    )
    
    # Process all patients
    processed_features = {}
    
    for idx, row in dataframe.iterrows():
        patient_id = row["PatientID"]
        patient_row_idx = dataframe.index.get_loc(idx)
        
        # Get standardized age for this patient
        age_features = age_scaled[patient_row_idx].flatten()
        
        # Get one-hot encoded categorical features for this patient
        categorical_features = categorical_encoded.iloc[patient_row_idx].values
        
        # Combine all features
        complete_features = np.concatenate([age_features, categorical_features])
        processed_features[patient_id] = complete_features
    
    # Store preprocessors for inference
    preprocessors = {
        'age_scaler': age_scaler,
        'age_median': age_median,
        'categorical_columns': list(categorical_encoded.columns),
        'feature_names': ['Age'] + list(categorical_encoded.columns),
        'n_features': len(complete_features)
    }
    
    print(f"Preprocessed features shape: {len(complete_features)} features per patient")
    print(f"Age feature: 1 (standardized)")
    print(f"Categorical features: {len(categorical_encoded.columns)} (one-hot encoded)")
    print(f"Feature breakdown:")
    for i, feature_name in enumerate(preprocessors['feature_names']):
        print(f"  {i}: {feature_name}")
    
    return {
        'features': processed_features,
        'preprocessors': preprocessors
    }

def load_and_cache_dataset(csv_path, image_directories, cache_path="cached_hecktor_data.pkl"):
    """Load and cache all data for faster access. Now also saves preprocessors for inference."""
    if os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}...")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    
    print("Loading and preprocessing dataset...")
    clinical_df = pd.read_csv(csv_path)
    
    # Set up image transforms
    image_transforms = create_image_transforms()
    
    # Load all medical images
    patient_images = {}
    failed_loads = []
    
    for idx, row in tqdm(clinical_df.iterrows(), total=len(clinical_df), desc="Loading images"):
        patient_id = row["PatientID"]
        
        try:
            ct_path = find_image_path(patient_id, "CT", image_directories)
            pet_path = find_image_path(patient_id, "PT", image_directories)
            
            transformed_data = image_transforms({"ct": ct_path, "pet": pet_path})
            combined_image = torch.cat([transformed_data["ct"], transformed_data["pet"]], dim=0)
            patient_images[patient_id] = combined_image
            
        except Exception as e:
            print(f"Failed to load images for {patient_id}: {e}")
            failed_loads.append(patient_id)
    
    # Remove failed loads from clinical data
    if failed_loads:
        print(f"Excluding {len(failed_loads)} patients due to missing images")
        clinical_df = clinical_df[~clinical_df["PatientID"].isin(failed_loads)]
    
    # Process clinical features (now uses one-hot encoding)
    clinical_features = preprocess_clinical_data(clinical_df)
    
    # Save clinical preprocessors separately for inference
    preprocessors_path = cache_path.replace('.pkl', '_clinical_preprocessors.pkl')
    with open(preprocessors_path, 'wb') as f:
        pickle.dump(clinical_features['preprocessors'], f)
    print(f"Clinical preprocessors saved to {preprocessors_path}")
    
    # Prepare survival data
    survival_outcomes = {}
    for idx, row in clinical_df.iterrows():
        patient_id = row["PatientID"]
        if patient_id in patient_images:
            survival_outcomes[patient_id] = {
                'time': float(row["RFS"]),
                'event': int(row["Relapse"])
            }
    
    # Package data
    cached_dataset = {
        'images': patient_images,
        'clinical_features': clinical_features,
        'survival_data': survival_outcomes,
        'dataframe': clinical_df[clinical_df["PatientID"].isin(patient_images.keys())]
    }
    
    # Save cache
    print(f"Saving preprocessed data to {cache_path}...")
    with open(cache_path, 'wb') as f:
        pickle.dump(cached_dataset, f)
    
    print(f"Successfully processed {len(patient_images)} patients")
    print(f"Each patient has {clinical_features['preprocessors']['n_features']} clinical features")
    return cached_dataset

# =============================================================================
# Main Function
# =============================================================================

def main():
    set_random_seed(RANDOM_SEED)
    os.makedirs("prognosis_logs", exist_ok=True)
    device = setup_device()

    print("Loading dataset...")
    dataset_cache = load_and_cache_dataset(
        csv_path=EHR_DATA_PATH,
        image_directories=[TRAINING_IMAGES_PATH],
        cache_path="hecktor_cache.pkl"
    )
    print(f"Total patients: {len(dataset_cache['images'])}")

    # Three-way split: 70% train / 15% val / 15% test
    clinical_df = dataset_cache['dataframe']
    all_pids = clinical_df["PatientID"].values
    all_events = clinical_df["Relapse"].values

    pids_trainval, pids_test, ev_trainval, _ = train_test_split(
        all_pids, all_events, test_size=0.15, stratify=all_events, random_state=RANDOM_SEED
    )
    pids_train, pids_val = train_test_split(
        pids_trainval, test_size=0.15 / 0.85, stratify=ev_trainval, random_state=RANDOM_SEED
    )
    print(f"Split — train: {len(pids_train)}, val: {len(pids_val)}, test: {len(pids_test)}")

    train_loader = DataLoader(HecktorSurvivalDataset(dataset_cache, pids_train),
                              batch_size=6, shuffle=True)
    val_loader   = DataLoader(HecktorSurvivalDataset(dataset_cache, pids_val),
                              batch_size=6, shuffle=False)
    test_loader  = DataLoader(HecktorSurvivalDataset(dataset_cache, pids_test),
                              batch_size=6, shuffle=False)

    clinical_dim = HecktorSurvivalDataset(dataset_cache, pids_train)[0][1].shape[0]
    model = HecktorSurvivalModel(clinical_feature_dim=clinical_dim, device=device, feature_dim=256)

    model.fit(train_loader=train_loader, val_loader=val_loader, num_epochs=100)

    val_c_index  = model.evaluate(val_loader)
    test_c_index = model.evaluate(test_loader)
    print(f"\nVal  C-index: {val_c_index:.4f}")
    print(f"Test C-index: {test_c_index:.4f}")

    model.save_model("prognosis_logs/final_model")

    test_predictions, test_times, test_events = model.predict(test_loader)
    pd.DataFrame({
        'patient_id':      pids_test,
        'risk_score':      test_predictions,
        'survival_time':   test_times,
        'event_indicator': test_events,
    }).to_csv("prognosis_logs/test_predictions.csv", index=False)


if __name__ == "__main__":
    main()