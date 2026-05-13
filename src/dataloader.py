import os
import random
from monai.data import DataLoader, CacheDataset
from transforms import get_train_transforms, get_validation_transforms
from typing import Tuple


def get_dataloaders(config, val_split: float = 0.2, seed: int = 42) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders with a simple random split.
    """
    images_dir = os.path.join(config.data_root, config.train_images_dir)
    ct_files = sorted(f for f in os.listdir(images_dir) if f.endswith("_ct.npz"))
    case_ids = [f.replace("_ct.npz", "") for f in ct_files]

    random.seed(seed)
    random.shuffle(case_ids)
    n_val = int(len(case_ids) * val_split)
    val_ids = case_ids[:n_val]
    train_ids = case_ids[n_val:]
    print(f"Split: {len(train_ids)} training cases, {len(val_ids)} validation cases")

    def make_file_list(ids):
        return [
            {
                "ct":    os.path.join(config.data_root, config.train_images_dir, f"{cid}_ct.npz"),
                "pet":   os.path.join(config.data_root, config.train_images_dir, f"{cid}_pet.npz"),
                "label": os.path.join(config.data_root, config.train_labels_dir,  f"{cid}_label.npz"),
            }
            for cid in ids
        ]

    train_files = make_file_list(train_ids)
    val_files   = make_file_list(val_ids)

    train_ds = CacheDataset(
        data=train_files,
        transform=get_train_transforms(config),
        cache_rate=config.cache_rate,
        num_workers=config.num_workers,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )

    val_ds = CacheDataset(
        data=val_files,
        transform=get_validation_transforms(),
        cache_rate=config.cache_rate,
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        persistent_workers=True,
    )

    return train_loader, val_loader
