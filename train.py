#!/usr/bin/env python3
"""Point d'entrée unique : entraînement end-to-end multitâche HECKTOR 2026."""

import os
import sys
import argparse
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _root)

from config import MultitaskConfig
from src.model import MultitaskModel
from src.dataset import get_multitask_dataloaders
from utils.losses import seg_loss, t_loss, n_loss, DeepHitDiscreteLoss, UncertaintyWeightedLoss as UncertaintyWeighting
from utils.metrics import (
    balanced_accuracy,
    discrete_risk_from_surv_logits,
    c_index,
)
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from monai.data import decollate_batch


def freeze(module: torch.nn.Module, frozen: bool):
    for p in module.parameters():
        p.requires_grad = not frozen


def compute_bin_edges(train_df, n_bins: int) -> np.ndarray:
    """Quantiles sur les temps d'événement du train set → T intervalles équiprobables."""
    ev = train_df[train_df["Relapse"] == 1]["RFS"].dropna().values.astype(float)
    if len(ev) < n_bins:
        # fallback : quantiles sur tous les temps
        ev = train_df["RFS"].dropna().values.astype(float)
    if len(ev) == 0:
        return np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(ev, np.linspace(0, 1, n_bins + 1))
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def train_one_epoch(model, loader, optimizer, weighting, deephit, bin_edges,
                    device, config, warmup: bool):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in tqdm(loader, desc="Train", leave=False):
        ct_pet   = batch["image"].to(device, non_blocking=True)
        seg_gt   = batch["label"].to(device, non_blocking=True)
        clinical = batch["clinical"].to(device, non_blocking=True).float()
        t_lbl    = batch["t_label"].to(device, non_blocking=True)
        n_lbl    = batch["n_label"].to(device, non_blocking=True)
        times    = batch["time"].to(device, non_blocking=True)
        events   = batch["event"].to(device, non_blocking=True)

        out = model(ct_pet, clinical)

        l_seg = seg_loss(out["seg_mask"], seg_gt)
        l_t   = t_loss(out["t_logits"], t_lbl)
        l_n   = n_loss(out["n_logits"], n_lbl)
        l_srv = deephit(out["surv_logits"], times, events, bin_edges)

        if warmup:
            loss = l_seg
        else:
            loss, _ = weighting(l_seg, l_t, l_n, l_srv)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip_norm)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(model, loader, deephit, bin_edges, device, config):
    model.eval()
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    post_label = AsDiscrete(to_onehot=config.num_seg_classes)
    post_pred  = AsDiscrete(argmax=True, to_onehot=config.num_seg_classes)

    t_true, t_pred = [], []
    n_true, n_pred = [], []
    risks, all_t, all_e = [], [], []

    for batch in tqdm(loader, desc="Val", leave=False):
        ct_pet   = batch["image"].to(device)
        seg_gt   = batch["label"].to(device)
        clinical = batch["clinical"].to(device).float()
        out = model(ct_pet, clinical)

        # Dice
        gt_list  = [post_label(x) for x in decollate_batch(seg_gt)]
        pr_list  = [post_pred(x)  for x in decollate_batch(out["seg_mask"])]
        dice_metric(y_pred=pr_list, y=gt_list)

        # T / N
        t_true.extend(batch["t_label"].numpy().tolist())
        t_pred.extend(out["t_logits"].argmax(1).cpu().numpy().tolist())
        n_true.extend(batch["n_label"].numpy().tolist())
        n_pred.extend(out["n_logits"].argmax(1).cpu().numpy().tolist())

        # Survie
        r = discrete_risk_from_surv_logits(out["surv_logits"]).cpu().numpy()
        risks.extend(r.tolist())
        all_t.extend(batch["time"].numpy().tolist())
        all_e.extend(batch["event"].numpy().tolist())

    dice = dice_metric.aggregate().item()
    dice_metric.reset()

    bal_t = balanced_accuracy(np.array(t_true), np.array(t_pred))
    bal_n = balanced_accuracy(np.array(n_true), np.array(n_pred))
    ci = c_index(np.array(risks), np.array(all_t), np.array(all_e))

    return {"dice": dice, "bal_t": bal_t, "bal_n": bal_n, "c_index": ci}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--cuda-device", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    config = MultitaskConfig()

    if config.device == "cuda" and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.cuda_device}")
        torch.cuda.set_device(device)
        print(f"[Device] {device} — {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cpu")
        print(f"[Device] {device}")

    # ---- Data ----
    train_loader, val_loader, train_df, _clin_enc = get_multitask_dataloaders(config)
    bin_edges_np = compute_bin_edges(train_df, config.n_time_bins)
    bin_edges = torch.tensor(bin_edges_np, dtype=torch.float32, device=device)
    print(f"[Data] bin edges (T={config.n_time_bins}) : {bin_edges_np}")

    # ---- Model ----
    model = MultitaskModel(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] {n_params:,} paramètres")

    # ---- Pertes / pondération ----
    weighting = UncertaintyWeighting(n_tasks=4).to(device)
    deephit   = DeepHitDiscreteLoss(alpha=0.2).to(device)

    optimizer = optim.AdamW(
        list(model.parameters()) + list(weighting.parameters()),
        lr=config.learning_rate, weight_decay=config.weight_decay,
    )
    scheduler = optim.lr_scheduler.PolynomialLR(
        optimizer, total_iters=config.num_epochs, power=config.poly_lr_power,
    )

    writer = SummaryWriter(config.log_dir) if config.use_tensorboard else None

    # ---- Resume ----
    start_epoch = 0
    best_metric = 0.0
    if args.resume:
        ckpt = model.load_checkpoint(args.resume, device)
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", -1) + 1
        best_metric = ckpt.get("best_metric", 0.0)
        print(f"[Resume] epoch={start_epoch}  best={best_metric:.4f}")

    # ---- Boucle ----
    for epoch in range(start_epoch, config.num_epochs):
        warmup = epoch < config.n_warmup

        # Gel / dégel des têtes T/N/Survie + cross-attn pendant le warm-up
        for mod in [model.t_head, model.n_head, model.surv_head,
                    model.cross_attn, model.clin_mlp, model.proj_tn]:
            freeze(mod, frozen=warmup)
        model.cls_token.requires_grad = not warmup

        print(f"\n=== Epoch {epoch+1}/{config.num_epochs} {'[warm-up seg]' if warmup else ''} ===")
        train_loss = train_one_epoch(
            model, train_loader, optimizer, weighting, deephit, bin_edges,
            device, config, warmup,
        )
        print(f"Train loss : {train_loss:.4f}")
        if writer:
            writer.add_scalar("Loss/train", train_loss, epoch)

        # ---- Validation ----
        should_val = (epoch + 1) % 5 == 0 or (epoch + 1) == config.num_epochs
        if should_val:
            metrics = validate(model, val_loader, deephit, bin_edges, device, config)
            print(f"Val   : Dice={metrics['dice']:.4f}  "
                  f"BalAcc T={metrics['bal_t']:.4f}  N={metrics['bal_n']:.4f}  "
                  f"C-index={metrics['c_index']:.4f}")
            if writer:
                for k, v in metrics.items():
                    writer.add_scalar(f"Val/{k}", v, epoch)

            # Métrique principale : moyenne pondérée (poids 1/4 par tâche, dice compte double)
            combined = 0.4 * metrics["dice"] + 0.2 * metrics["bal_t"] \
                       + 0.2 * metrics["bal_n"] + 0.2 * metrics["c_index"]
            if combined > best_metric:
                best_metric = combined
                path = os.path.join(config.checkpoint_dir, "best_model.pth")
                model.save_checkpoint(path, epoch, optimizer.state_dict(),
                                      best_metric=best_metric)
                print(f"[Save] nouveau meilleur modèle ({combined:.4f}) → {path}")

        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        if writer:
            writer.add_scalar("LR", lr, epoch)

        if config.save_checkpoint_every > 0 and (epoch + 1) % config.save_checkpoint_every == 0:
            path = os.path.join(config.checkpoint_dir, "last_model.pth")
            model.save_checkpoint(path, epoch, optimizer.state_dict(),
                                  best_metric=best_metric)

    if writer:
        writer.close()
    print("\n✔ Entraînement terminé.")


if __name__ == "__main__":
    main()
