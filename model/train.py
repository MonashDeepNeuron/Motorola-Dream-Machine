#!/usr/bin/env python3
# ===============================================================
# EEG-to-Robot-Arm — Training script (lean)
# - Loads X.npy (N,C,F,T) and y.npy (N,)
# - Trains EEG2Arm and saves a robust checkpoint (*.pt)
# ===============================================================

import argparse
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset

from eeg_model import EEG2Arm, safe_compile          # your model
from utils import (                       # shared utils
    ModelCfg, set_seed, save_checkpoint
)

# tqdm is optional
try:
    from tqdm import tqdm
except Exception:
    tqdm = lambda x, **k: x  # fallback


# ------------------------
# Dataset: X.npy / y.npy
# ------------------------
class EEGNpyDataset(Dataset):
    """
    Expects:
      - X_path: npy array of shape (N, C, F, T), float32/float64
      - y_path: npy array of shape (N,), int labels in [0..K-1]
    Optional:
      - per_sample_normalize: z-score each sample over (F, T) per electrode
    """
    def __init__(self, X_path: str, y_path: str, mmap: bool = True, per_sample_normalize: bool = False, eps: float = 1e-5):
        self.X = np.load(X_path, mmap_mode="r" if mmap else None)
        self.y = np.load(y_path)
        if self.X.ndim != 4:
            raise ValueError(f"X must be (N, C, F, T), got {self.X.shape}")
        if self.y.ndim != 1:
            raise ValueError(f"y must be (N,), got {self.y.shape}")
        if self.X.shape[0] != self.y.shape[0]:
            raise ValueError(f"N mismatch: {self.X.shape[0]} vs {self.y.shape[0]}")
        self.per_sample_normalize = per_sample_normalize
        self.eps = eps

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        x = self.X[i].astype(np.float32, copy=False)  # (C, F, T)
        y = int(self.y[i])
        x_t = torch.from_numpy(x)

        if self.per_sample_normalize:
            mu = x_t.mean(dim=(1, 2), keepdim=True)
            sd = x_t.std(dim=(1, 2), keepdim=True)
            x_t = (x_t - mu) / (sd + self.eps)

        return x_t, y


def make_loaders(
    X_path: str,
    y_path: str,
    batch_size: int = 64,
    val_split: float = 0.2,
    num_workers: int = 2,
    seed: int = 1337,
    per_sample_normalize: bool = False,
):
    ds = EEGNpyDataset(X_path, y_path, mmap=True, per_sample_normalize=per_sample_normalize)
    n = len(ds)
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)

    n_val = int(round(val_split * n))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    ds_train = Subset(ds, train_idx.tolist())
    ds_val = Subset(ds, val_idx.tolist())

    train_loader = DataLoader(
        ds_train, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(), drop_last=False
    )
    val_loader = DataLoader(
        ds_val, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(), drop_last=False
    )

    C, F, T = ds[0][0].shape
    return train_loader, val_loader, (C, F, T)


# ------------------------
# Training / Eval
# ------------------------
def train_one_epoch(model, loader, optimizer, scaler, device, criterion, max_grad_norm: float = 1.0):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for x, y in tqdm(loader, desc="train", leave=False):
        x = x.to(device, non_blocking=True)         # (B, C, F, T)
        y = torch.as_tensor(y, device=device, dtype=torch.long)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            logits = model(x)                        # (B, K)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        if max_grad_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        total_loss += float(loss.item()) * x.size(0)
        preds = torch.argmax(logits, dim=-1)
        total_correct += int((preds == y).sum().item())
        total_count += x.size(0)

    return total_loss / max(1, total_count), total_correct / max(1, total_count)


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for x, y in tqdm(loader, desc="valid", leave=False):
        x = x.to(device, non_blocking=True)
        y = torch.as_tensor(y, device=device, dtype=torch.long)
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            logits = model(x)
            loss = criterion(logits, y)
        total_loss += float(loss.item()) * x.size(0)
        preds = torch.argmax(logits, dim=-1)
        total_correct += int((preds == y).sum().item())
        total_count += x.size(0)

    return total_loss / max(1, total_count), total_correct / max(1, total_count)


# ------------------------
# Fit
# ------------------------
def fit(
    X_path: str,
    y_path: str,
    out_ckpt: str,
    classes: int,
    n_elec: int,
    n_bands: int,
    cnn_time_pool: int = 2,
    frame_embed: int = 256,
    p_drop: float = 0.1,
    batch_size: int = 64,
    epochs: int = 30,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    val_split: float = 0.2,
    seed: int = 1337,
    compile_on_cuda: bool = False,
    per_sample_normalize: bool = False,
    num_workers: int = 2,
):
    from utils import save_checkpoint  # local import is ok

    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Data
    train_loader, val_loader, (C, F, T) = make_loaders(
        X_path, y_path, batch_size=batch_size, val_split=val_split,
        num_workers=num_workers, seed=seed, per_sample_normalize=per_sample_normalize
    )

    if C != n_elec or F != n_bands:
        raise ValueError(f"Data shapes (C={C}, F={F}) do not match config (n_elec={n_elec}, n_bands={n_bands}).")

    # Model
    model = EEG2Arm(
        n_elec=n_elec,
        n_bands=n_bands,
        clip_length=None,
        cnn_time_pool=cnn_time_pool,
        frame_embed=frame_embed,
        n_classes=classes,
        p_drop=p_drop,
    ).to(device)

    if compile_on_cuda:
        model = safe_compile(model)

    # Loss/optim
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, verbose=True)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    best_val = float("inf")
    best_epoch = -1

    # Materialize LazyLinear
    with torch.no_grad():
        for x0, _ in train_loader:
            _ = model(x0.to(device)[:1])
            break

    # Train
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, scaler, device, criterion)
        va_loss, va_acc = evaluate(model, val_loader, device, criterion)
        scheduler.step(va_loss)

        print(f"[epoch {epoch:03d}] train: loss={tr_loss:.4f} acc={tr_acc:.3f} | val: loss={va_loss:.4f} acc={va_acc:.3f}")

        if va_loss < best_val:
            best_val = va_loss
            best_epoch = epoch
            cfg = ModelCfg(
                n_elec=n_elec, n_bands=n_bands, clip_length=None,
                cnn_time_pool=cnn_time_pool, frame_embed=frame_embed,
                n_classes=classes, p_drop=p_drop
            )
            example_input_shape = (1, n_elec, n_bands, T)
            metrics = {"val_loss": va_loss, "val_acc": va_acc}
            save_checkpoint(out_ckpt, model, cfg, example_input_shape, epoch, metrics)

    print(f"[done] best epoch={best_epoch}, best val_loss={best_val:.4f}")


# ------------------------
# CLI
# ------------------------
def build_argparser():
    p = argparse.ArgumentParser(description="EEG2Arm — train model and save checkpoint")
    p.add_argument("--x", required=True, help="Path to X.npy (N,C,F,T)")
    p.add_argument("--y", required=True, help="Path to y.npy (N,)")
    p.add_argument("--save", required=True, help="Output checkpoint path (*.pt)")
    p.add_argument("--classes", type=int, required=True, help="Number of classes (K)")

    p.add_argument("--elec", type=int, required=True, help="Electrodes (C)")
    p.add_argument("--bands", type=int, required=True, help="Bands (F)")
    p.add_argument("--cnn-time-pool", type=int, default=2)
    p.add_argument("--frame-embed", type=int, default=256)
    p.add_argument("--drop", type=float, default=0.1)

    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--val-split", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--normalize", action="store_true", help="Per-sample z-score over (F,T)")
    p.add_argument("--compile", action="store_true", help="Try torch.compile on CUDA")
    return p


def main():
    ap = build_argparser()
    args = ap.parse_args()

    fit(
        X_path=args.x,
        y_path=args.y,
        out_ckpt=args.save,
        classes=args.classes,
        n_elec=args.elec,
        n_bands=args.bands,
        cnn_time_pool=args.cnn_time_pool,
        frame_embed=args.frame_embed,
        p_drop=args.drop,
        batch_size=args.batch,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.wd,
        val_split=args.val_split,
        seed=args.seed,
        compile_on_cuda=args.compile,
        per_sample_normalize=args.normalize,
        num_workers=args.workers,
    )


if __name__ == "__main__":
    main()
