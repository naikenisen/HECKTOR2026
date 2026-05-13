"""Data transforms for HECKTOR dataset, adapted for preprocessed .npz files."""

import numpy as np
import torch
from monai.transforms import (
    Compose,
    MapTransform,
    EnsureChannelFirstd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    EnsureTyped,
    RandCropByLabelClassesd,
    RandCropByPosNegLabeld,
    ConcatItemsd,
    SelectItemsd,
)

class LoadNpzDictd(MapTransform):
    """
    Custom transform to load data from .npz files.
    Each .npz file is expected to contain an 'image' array and a 'meta' dictionary.
    This transform correctly loads the data and reconstructs it into a
    MONAI-compatible format with a tensor and its associated metadata dictionary.
    """
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            filepath = d[key]
            # Use allow_pickle=True for loading the metadata dictionary.
            # This is required for security reasons in recent NumPy versions.
            loaded = np.load(filepath, allow_pickle=True)
            
            image_array = loaded['image']
            meta_dict = loaded['meta'].item()  # .item() extracts the dict from the array

            # Reconstruct the data in a MONAI-friendly format.
            # Create a PyTorch tensor from the numpy array.
            d[key] = torch.from_numpy(image_array)
            
            # MONAI transforms expect metadata in a separate dict with the format {key}_meta_dict.
            d[f"{key}_meta_dict"] = meta_dict
        return d


def get_multitask_train_transforms(config):
    """
    Variante multitâche : conserve clinical/t_label/n_label/time/event/case_id.
    `RandCropByLabelClassesd` est conservé pour échantillonner près des tumeurs.
    """
    keys = ["ct", "pet", "label"]
    keep = ["image", "label", "clinical", "t_label", "n_label", "time", "event", "case_id"]

    transforms = [LoadNpzDictd(keys=keys)]

    transforms.append(
        RandCropByLabelClassesd(
            keys=keys,
            label_key="label",
            spatial_size=config.spatial_size,
            ratios=[0.1, 0.45, 0.45],
            num_classes=3,
            num_samples=1,        # 1 sample/batch item pour préserver l'alignement tabulaire
            allow_missing_keys=True,
            warn=False,
        )
    )

    if config.use_augmentation:
        transforms.extend([
            RandFlipd(keys=keys, spatial_axis=[0, 1, 2], prob=config.aug_probability),
            RandScaleIntensityd(keys=["ct"], factors=0.1, prob=config.aug_probability),
            RandShiftIntensityd(keys=["ct"], offsets=0.1, prob=config.aug_probability),
            RandGaussianNoised(keys=["ct"], std=0.01, prob=config.aug_probability),
            RandGaussianSmoothd(
                keys=["ct"],
                sigma_x=(0.5, 1.15), sigma_y=(0.5, 1.15), sigma_z=(0.5, 1.15),
                prob=config.aug_probability,
            ),
        ])

    transforms.extend([
        ConcatItemsd(keys=["ct", "pet"], name="image", dim=0),
        SelectItemsd(keys=keep),
        EnsureTyped(keys=["image", "label"]),
    ])
    return Compose(transforms)


def get_multitask_validation_transforms():
    """Variante multitâche pour la validation : pas de crop aléatoire."""
    keys = ["ct", "pet", "label"]
    keep = ["image", "label", "clinical", "t_label", "n_label", "time", "event", "case_id"]
    return Compose([
        LoadNpzDictd(keys=keys),
        ConcatItemsd(keys=["ct", "pet"], name="image", dim=0),
        SelectItemsd(keys=keep),
        EnsureTyped(keys=["image", "label"]),
    ])


