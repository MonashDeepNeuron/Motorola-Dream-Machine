#!/usr/bin/env python3
# ===============================================================
# v1.4
# EEG-to-Robot-Arm — CNN (freq/time) → early spatial mix → Tiny Transformer
# - No GCN
# - Device/dtype-safe positional encoding
# - LazyLinear-safe init
# - Forward that returns final logits AND a forward_seq for per-frame logits
# - Hysteresis helper + micro-benchmark + smoke test
# ===============================================================

from __future__ import annotations

import math
import time
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import UninitializedParameter

# Optional perf toggles
try:
    torch.set_float32_matmul_precision("medium")
except Exception:
    pass
torch.backends.cudnn.benchmark = True


# ----------------------------------------------------------------
# Positional encoding (dtype/device safe)
# ----------------------------------------------------------------
class PositionalEncoding(nn.Module):
    """
    Classic sinusoidal positional encoding for (B, T, d).
    Stored in float32; cast to input dtype/device on forward.
    """
    def __init__(self, d_model: int, max_len: int = 10_000):
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # (T,1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10_000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)  # even
        pe[:, 1::2] = torch.cos(position * div_term)  # odd
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # (1, T, d)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, T, d)
        return x + self.pe[:, : x.size(1)].to(dtype=x.dtype, device=x.device)


# ----------------------------------------------------------------
# CNN stem blocks
# ----------------------------------------------------------------
class DWConvBlock(nn.Module):
    """
    Depth-wise 3D conv over (freq,time) per electrode.
    Input  : (B, C, D=1, F, T)   (C = electrodes, D is a dummy)
    Output : (B, C, 1, F', T')   (pool only over F/T, never over C)
    """
    def __init__(
        self,
        in_ch: int,
        k_freq: int = 3,
        k_time: int = 3,
        pool: Optional[Tuple[int, int, int]] = (1, 2, 2),
        p_drop: float = 0.0,
    ):
        super().__init__()
        self.dw = nn.Conv3d(
            in_ch,
            in_ch,
            kernel_size=(1, k_freq, k_time),
            padding=(0, k_freq // 2, k_time // 2),
            groups=in_ch,  # per-electrode
            bias=False,
        )
        self.bn = nn.BatchNorm3d(in_ch)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool3d(pool) if pool else nn.Identity()
        self.drop = nn.Dropout3d(p_drop) if p_drop > 0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = self.dw(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.pool(x)
        x = self.drop(x)
        return x


class PointwiseMix(nn.Module):
    """
    1x1x1 Conv3d for spatial mixing across electrodes.
    groups=1 → full mixing across electrodes (intended when not using a GCN).
    """
    def __init__(self, in_ch: int, out_ch: int, groups: int = 1, p_drop: float = 0.0):
        super().__init__()
        self.pw = nn.Conv3d(in_ch, out_ch, kernel_size=1, groups=groups, bias=False)
        self.bn = nn.BatchNorm3d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout3d(p_drop) if p_drop > 0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = self.pw(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.drop(x)
        return x


# ----------------------------------------------------------------
# Tiny causal Transformer over frames
# ----------------------------------------------------------------
class TinyTransformer(nn.Module):
    """
    2-layer causal Transformer over time frames.
    Input:  (B, T, E)
    Output: (B, E)  ← last time step embedding
    """
    def __init__(
        self, embed: int, n_heads: int = 4, ff: int = 512, n_layers: int = 2, p_drop: float = 0.1
    ):
        super().__init__()
        # norm_first=False to avoid nested-tensor warning; functional difference is negligible here
        layer = nn.TransformerEncoderLayer(
            d_model=embed,
            nhead=n_heads,
            dim_feedforward=ff,
            dropout=p_drop,
            activation="relu",
            batch_first=True,
            norm_first=False,
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.pos = PositionalEncoding(embed)

    def forward(self, x: Tensor) -> Tensor:
        # causal mask: prevent future leakage
        B, T, E = x.shape
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        z = self.enc(self.pos(x), mask=mask)
        return z[:, -1]  # (B, E)


# ----------------------------------------------------------------
# Complete model: CNN → (early mix) → per-frame proj → Transformer → Head
# ----------------------------------------------------------------
class EEG2Arm(nn.Module):
    """
    EEG → DW-3D CNN (freq/time per electrode) → 1x1x1 spatial mix (groups=1)
        → flatten per frame → Linear projection → tiny causal Transformer → Head

    Args:
      n_elec         : electrodes (e.g., 32)
      n_bands        : bands per channel (use 4 for Δ/α/β/γ)
      cnn_time_pool  : pooling factor along time in the first DW block (e.g., 2)
      frame_embed    : embedding dim given to Transformer (e.g., 256)
      n_classes      : number of output classes
      p_drop         : dropout probability
    """
    def __init__(
        self,
        n_elec: int = 32,
        n_bands: int = 4,
        clip_length: Optional[int] = None,  # optional assert on T
        cnn_time_pool: int = 2,
        frame_embed: int = 256,
        n_classes: int = 5,
        p_drop: float = 0.1,
    ):
        super().__init__()
        self.n_elec = n_elec
        self.clip_length = clip_length

        # ----- CNN stem -----
        # Input to conv3d will be (B, C=n_elec, D=1, F=n_bands, T)
        self.dw1 = DWConvBlock(n_elec, pool=(1, 2, cnn_time_pool), p_drop=p_drop)
        self.dw2 = DWConvBlock(n_elec, pool=None, p_drop=p_drop)

        # Early spatial mixing across electrodes (groups=1)
        self.mix = PointwiseMix(in_ch=n_elec, out_ch=2 * n_elec, groups=1, p_drop=p_drop)

        # Per-frame projection to Transformer embed size
        # Input feature per frame after stem = (2C * F'), where F' depends on pooling.
        # Use LazyLinear so it infers the correct input size on first forward.
        self.frame_proj = nn.Sequential(
            nn.LazyLinear(frame_embed),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
        )

        # Temporal model
        self.tform = TinyTransformer(embed=frame_embed, n_heads=4, ff=512, n_layers=2, p_drop=p_drop)

        # Head (works for both (B, E) and (B, T, E) — nn.Linear applies on last dim)
        self.fc = nn.Sequential(
            nn.Linear(frame_embed, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
            nn.Linear(128, n_classes),
        )

        self.apply(self._init_weights)

        # Guard-rails to ensure we never pool/stride across the electrode axis
        assert self.dw1.dw.kernel_size[0] == 1 and self.dw1.dw.stride[0] == 1
        assert self.dw2.dw.kernel_size[0] == 1 and self.dw2.dw.stride[0] == 1

    # ---- Safe initializer (skip lazy params until after first forward) ----
    @staticmethod
    def _init_weights(m: nn.Module):
        w = getattr(m, "weight", None)
        b = getattr(m, "bias", None)
        if isinstance(w, UninitializedParameter) or isinstance(b, UninitializedParameter):
            return
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if b is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm)):
            if hasattr(m, "weight") and m.weight is not None and not isinstance(m.weight, UninitializedParameter):
                nn.init.ones_(m.weight)
            if hasattr(m, "bias") and m.bias is not None and not isinstance(m.bias, UninitializedParameter):
                nn.init.zeros_(m.bias)

    # ---- Frame-by-frame logits (useful for plotting & streaming smoothing) ----
    def forward_seq(self, x: Tensor) -> Tensor:
        """
        x : (B, n_elec, n_bands, T)
        returns logits for each post-stem frame: (B, T2, n_classes)
        """
        B, C, F0, T0 = x.shape
        if C != self.n_elec:
            raise ValueError(f"expected {self.n_elec} electrodes, got {C}")
        if self.clip_length is not None and T0 != self.clip_length:
            raise ValueError(f"expected clip_length={self.clip_length}, got {T0}")

        # CNN stem
        x = x.unsqueeze(2)                 # (B, C, 1, F0, T0)
        x = self.dw1(x)                    # (B, C, 1, F1, T1)
        x = self.dw2(x)                    # (B, C, 1, F2, T2)
        x = self.mix(x)                    # (B, 2C, 1, F2, T2)
        x = x.squeeze(2)                   # (B, 2C, F2, T2)

        # Per-frame flatten → projection
        x = x.permute(0, 3, 1, 2).contiguous()  # (B, T2, 2C, F2)
        B, T2, C2, F2 = x.shape
        x = x.view(B, T2, C2 * F2)              # (B, T2, 2C*F2)
        x = self.frame_proj(x)                  # (B, T2, E)

        # Transformer (causal) → per-frame embeddings
        mask = torch.triu(torch.ones(T2, T2, device=x.device, dtype=torch.bool), diagonal=1)
        z = self.tform.enc(self.tform.pos(x), mask=mask)  # (B, T2, E)

        # Per-frame logits
        logits = self.fc(z)                     # (B, T2, n_classes)
        return logits

    # ---- Final-only logits (last frame) ----
    def forward(self, x: Tensor) -> Tensor:
        """
        x : (B, n_elec, n_bands, T)
        returns logits for the latest frame: (B, n_classes)
        """
        logits_seq = self.forward_seq(x)        # (B, T2, K)
        return logits_seq[:, -1, :]             # (B, K)


# ----------------------------------------------------------------
# Decision smoothing (hysteresis)
# ----------------------------------------------------------------
class Hysteresis:
    """
    Simple N-consecutive-frames hysteresis for stable class decisions.
    """
    def __init__(self, k: int, prob_thresh: float = 0.75, consecutive: int = 3, rest_idx: int = 4):
        self.k = k
        self.p = prob_thresh
        self.n = consecutive
        self.rest = rest_idx
        self.counts = torch.zeros(k, dtype=torch.int32)
        self.curr = rest_idx

    def step(self, probs: Tensor) -> int:
        """
        probs: (K,) tensor of softmax probabilities (CPU or CUDA).
        Returns the smoothed class index.
        """
        if probs.is_cuda:
            probs = probs.detach().cpu()
        k_hat = int(torch.argmax(probs).item())
        if float(probs[k_hat].item()) >= self.p:
            self.counts.zero_()
            self.counts[k_hat] += 1
            if int(self.counts[k_hat].item()) >= self.n:
                self.curr = k_hat
        else:
            self.counts.zero_()
        return self.curr


# ----------------------------------------------------------------
# Micro-benchmark
# ----------------------------------------------------------------
def benchmark(model: nn.Module, shape=(1, 32, 4, 12), iters: int = 200, warmup: int = 20, device: str = "cpu"):
    x = torch.randn(*shape, device=device)
    model.eval()
    # warmup
    for _ in range(warmup):
        _ = model(x)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = model(x)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / iters * 1000.0
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[benchmark] {shape} → {ms:.2f} ms/call, params={n_params/1e6:.3f}M")


# ----------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    B, C, F, T = 4, 32, 4, 12  # 4 bands (Δ/α/β/γ)
    dummy = torch.randn(B, C, F, T)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = EEG2Arm(
        n_elec=C,
        n_bands=F,
        clip_length=None,  # accept any T
        cnn_time_pool=2,
        frame_embed=256,
        n_classes=5,
        p_drop=0.1,
    ).to(device)

    # (Optional) Torch 2.x: compile for extra speed
    if hasattr(torch, "compile"):
        try:
            model = torch.compile(model)  # type: ignore[attr-defined]
        except Exception:
            pass

    dummy = dummy.to(device)

    # Final-only logits
    out = model(dummy)
    print("out shape:", out.shape)  # (B, 5)

    # Per-frame logits
    out_seq = model.forward_seq(dummy)
    print("out_seq shape:", out_seq.shape)  # (B, T2, 5)

    # Benchmark
    benchmark(model, shape=(1, C, F, T), device=device)

    # Hysteresis demo (toy probabilities)
    hys = Hysteresis(k=5, prob_thresh=0.75, consecutive=3, rest_idx=4)
    probs = torch.tensor([0.1, 0.1, 0.1, 0.8, 0.1])  # pretend "grasp" is index 3
    for t in range(5):
        cls = hys.step(probs)
        print(f"[tick {t}] cls={cls}")

    torch.save(model.state_dict(), "eeg_model.pth")
