#!/usr/bin/env python3
"""
Inference script for HECKTOR survival prediction.
Usage: python prognosis_inference.py --csv test_data.csv --input_path ./test_images
       --model_path prognosis_logs/final_model_feature_extractor.pt
       --clinical_preprocessors hecktor_cache_clinical_preprocessors.pkl
"""

import argparse
import os
import numpy as np
import pandas as pd
import torch
import pickle
from torch.utils.data import DataLoader
from lifelines.utils import concordance_index

from prognosis import (
    FusedFeatureExtractor,
    HecktorSurvivalDataset,
    create_image_transforms,
    find_image_path,
    set_random_seed,
    RANDOM_SEED,
)


def load_and_preprocess_test_data(csv_path, input_path, clinical_preprocessors_path):
    test_df = pd.read_csv(csv_path)
    has_survival_data = 'RFS' in test_df.columns and 'Relapse' in test_df.columns

    with open(clinical_preprocessors_path, 'rb') as f:
        clinical_preprocessors = pickle.load(f)

    image_transforms = create_image_transforms()
    patient_images = {}
    failed_loads = []

    for _, row in test_df.iterrows():
        patient_id = row["PatientID"]
        try:
            ct_path  = find_image_path(patient_id, "CT", [input_path])
            pet_path = find_image_path(patient_id, "PT", [input_path])
            transformed = image_transforms({"ct": ct_path, "pet": pet_path})
            patient_images[patient_id] = torch.cat([transformed["ct"], transformed["pet"]], dim=0)
        except Exception:
            failed_loads.append(patient_id)

    if failed_loads:
        test_df = test_df[~test_df["PatientID"].isin(failed_loads)]

    # Preprocess clinical features using saved preprocessors
    ALL_CLINICAL_FEATURES  = ["Age", "Gender", "Tobacco Consumption", "Alcohol Consumption",
                               "Performance Status", "M-stage", "Treatment"]
    CATEGORICAL_FEATURES   = ["Gender", "Tobacco Consumption", "Alcohol Consumption",
                               "Performance Status", "M-stage", "Treatment"]

    feature_subset = test_df[ALL_CLINICAL_FEATURES].copy()
    age_scaled = clinical_preprocessors['age_scaler'].transform(
        feature_subset[["Age"]].fillna(clinical_preprocessors['age_median'])
    )

    cat_data = feature_subset[CATEGORICAL_FEATURES].astype(str).fillna('Unknown')
    cat_enc  = pd.get_dummies(cat_data, columns=CATEGORICAL_FEATURES,
                               prefix=CATEGORICAL_FEATURES, dummy_na=False, drop_first=False)
    for col in clinical_preprocessors['categorical_columns']:
        if col not in cat_enc.columns:
            cat_enc[col] = 0
    cat_enc = cat_enc[clinical_preprocessors['categorical_columns']].astype(np.float32)

    processed_features = {}
    for i, (_, row) in enumerate(test_df.iterrows()):
        pid = row["PatientID"]
        processed_features[pid] = np.concatenate(
            [age_scaled[i].flatten(), cat_enc.iloc[i].values]
        ).astype(np.float32)

    survival_outcomes = {}
    for _, row in test_df.iterrows():
        pid = row["PatientID"]
        if pid in patient_images:
            if has_survival_data:
                survival_outcomes[pid] = {'time': float(row["RFS"]), 'event': int(row["Relapse"])}
            else:
                survival_outcomes[pid] = {'time': 0.0, 'event': 0}

    valid_pids = [pid for pid in test_df["PatientID"] if pid in patient_images]
    return {
        'images':            patient_images,
        'clinical_features': {'features': processed_features},
        'survival_data':     survival_outcomes,
        'dataframe':         test_df[test_df["PatientID"].isin(valid_pids)],
        'has_survival_data': has_survival_data,
        'patient_ids':       valid_pids,
    }


def run_inference(csv_path, input_path, model_path, icare_model_path,
                  clinical_preprocessors_path, batch_size=4):
    set_random_seed(RANDOM_SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_data = load_and_preprocess_test_data(csv_path, input_path, clinical_preprocessors_path)
    patient_ids = test_data['patient_ids']

    dataset    = HecktorSurvivalDataset(test_data, patient_ids)
    loader     = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    clinical_dim = dataset[0][1].shape[0]

    # Load feature extractor
    feature_extractor = FusedFeatureExtractor(
        clinical_feature_dim=clinical_dim,
        feature_output_dim=128,
    ).to(device)
    feature_extractor.load_state_dict(torch.load(model_path, map_location=device))
    feature_extractor.eval()

    # Load BaggedIcareSurvival
    with open(icare_model_path, 'rb') as f:
        icare_model = pickle.load(f)

    # Extract features
    all_features = []
    with torch.no_grad():
        for images, clinical, *_ in loader:
            images, clinical = images.to(device), clinical.to(device)
            all_features.append(feature_extractor(images, clinical).cpu().numpy())
    features = np.vstack(all_features)

    predictions = icare_model.predict(features)

    # Evaluate if survival labels are available
    if test_data['has_survival_data']:
        times  = np.array([test_data['survival_data'][pid]['time']  for pid in patient_ids])
        events = np.array([test_data['survival_data'][pid]['event'] for pid in patient_ids])
        c_index = concordance_index(times, -predictions, events)
        print(f"C-index: {c_index:.4f}")

    return predictions, patient_ids


def main():
    parser = argparse.ArgumentParser(description="HECKTOR survival prediction inference")
    parser.add_argument("--csv",                      required=True)
    parser.add_argument("--input_path",               required=True)
    parser.add_argument("--model_path",               required=True,
                        help="Path to feature extractor checkpoint (.pt)")
    parser.add_argument("--icare_model_path",         required=True,
                        help="Path to BaggedIcareSurvival pickle (.pkl)")
    parser.add_argument("--clinical_preprocessors",   required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    for path in [args.csv, args.input_path, args.model_path,
                 args.icare_model_path, args.clinical_preprocessors]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Not found: {path}")

    run_inference(
        csv_path=args.csv,
        input_path=args.input_path,
        model_path=args.model_path,
        icare_model_path=args.icare_model_path,
        clinical_preprocessors_path=args.clinical_preprocessors,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
