#!/usr/bin/env python3
# ===============================================================
# v1.0
# Shared utilities for EEG2Arm:
# - ModelCfg (checkpoint config)
# - set_seed
# - save_checkpoint
# - load_model_from_checkpoint (handles LazyLinear materialization)
# - predict_npy (batch inference from X.npy)
# ===============================================================

from dataclasses import dataclass, asdict
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from eeg_model import EEG2Arm


# ------------------------
# Repro
# ------------------------
def set_seed(seed: int = 1337):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ------------------------
# Config stored in checkpoint
# ------------------------
@dataclass
class ModelCfg:
    n_elec: int
    n_bands: int
    clip_length: Optional[int]
    cnn_time_pool: int
    frame_embed: int
    n_classes: int
    p_drop: float


# ------------------------
# Lazy layer materialization
# ------------------------
@torch.no_grad()
def _materialize_lazy_layers(model: nn.Module, device: str, example_input_shape: Tuple[int, int, int, int]):
    """
    Runs a single forward pass to initialize LazyLinear parameters so state_dict can load.
    """
    B, C, F, T = example_input_shape
    dummy = torch.zeros(B, C, F, T, device=device, dtype=torch.float32)
    _ = model(dummy)


# ------------------------
# Checkpoint helpers
# ------------------------
def save_checkpoint(
    path: str,
    model: nn.Module,
    cfg: ModelCfg,
    example_input_shape: Tuple[int, int, int, int],
    epoch: int,
    metrics: dict,
):
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ckpt = {
        "model_state": model.state_dict(),
        "config": asdict(cfg),
        "example_input_shape": example_input_shape,  # e.g., (1, C, F, T)
        "epoch": int(epoch),
        "metrics": metrics,
        "version": "eeg2arm_v1.5",
    }
    torch.save(ckpt, path)
    print(f"[save] wrote checkpoint to: {path}")


def load_model_from_checkpoint(path: str, device: str = "cpu") -> Tuple[nn.Module, dict]:
    ckpt = torch.load(path, map_location=device)
    cfg = ModelCfg(**ckpt["config"])

    model = EEG2Arm(
        n_elec=cfg.n_elec,
        n_bands=cfg.n_bands,
        clip_length=cfg.clip_length,
        cnn_time_pool=cfg.cnn_time_pool,
        frame_embed=cfg.frame_embed,
        n_classes=cfg.n_classes,
        p_drop=cfg.p_drop,
    ).to(device)

    # Initialize LazyLinear before load
    example_shape = tuple(ckpt.get("example_input_shape", (1, cfg.n_elec, cfg.n_bands, 12)))
    _materialize_lazy_layers(model, device, example_shape)

    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    return model, ckpt


# ------------------------
# Batch prediction from X.npy
# ------------------------
@torch.no_grad()
def predict_npy(
    ckpt_path: str,
    X_path: str,
    apply_softmax: bool = True,
    per_sample_normalize: bool = False,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ckpt = load_model_from_checkpoint(ckpt_path, device=device)

    X = np.load(X_path)
    if X.ndim == 3:   # (C, F, T) -> (1, C, F, T)
        X = X[None, ...]
    if X.ndim != 4:
        raise ValueError(f"Expected input shape (B, C, F, T) or (C, F, T); got {X.shape}")

    X_t = torch.from_numpy(X.astype(np.float32, copy=False)).to(device)

    if per_sample_normalize:
        # vectorized z-score per sample/channel over (F,T)
        mu = X_t.mean(dim=(2, 3), keepdim=True)
        sd = X_t.std(dim=(2, 3), keepdim=True)
        X_t = (X_t - mu) / (sd + 1e-5)

    logits = model(X_t)  # (B, K)
    if apply_softmax:
        probs = torch.softmax(logits, dim=-1)
        top_p, top_i = probs.max(dim=-1)
        return top_i.cpu().numpy(), top_p.cpu().numpy(), probs.cpu().numpy()
    else:
        top_v, top_i = logits.max(dim=-1)
        return top_i.cpu().numpy(), top_v.cpu().numpy(), logits.cpu().numpy()
