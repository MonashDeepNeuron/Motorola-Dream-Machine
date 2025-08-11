# ===========================================
# EEG → Bandpowers (Δ, α, β, γ) → TCN → Class
# ===========================================
# Author: Joshua Chua (MDN)
# Notes:
# - This is a lean, real-time friendly TCN-only pipeline.
# - Feature window L=0.5 s (64 samples @128 Hz), hop=0.125 s (16 samples).
# - Inputs per tick: 32 channels × 4 bands = 128 features.
# - Sequence length to model T=10 (~1.25 s context).
# - Includes training scaffold and streaming inference with hysteresis.
# -------------------------------------------

import math
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Optional, Iterable

# Torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# SciPy for signal processing
from scipy.signal import iirnotch, butter, filtfilt, welch

# Utils
from tqdm import tqdm
torch.set_float32_matmul_precision('medium')

# ======== Signal Processing: filters and bandpowers ========

@dataclass
class FeatureConfig:
    fs: int = 128                  # sampling rate (Hz)
    window_s: float = 0.5          # analysis window (seconds)
    hop_s: float = 0.125           # hop (seconds)
    n_channels: int = 32
    bands: Tuple[Tuple[float,float], ...] = ((0.5,4.0),(8.0,12.0),(13.0,30.0),(30.0,40.0))
    notch_hz: float = 50.0
    bandpass: Tuple[float,float] = (0.5, 40.0)
    eps: float = 1e-8              # numerical stability
    use_relative_log: bool = True  # relative log-power
    welch_nperseg: Optional[int] = None  # defaults to window samples

    @property
    def window_samples(self) -> int:
        return int(self.window_s * self.fs)
    @property
    def hop_samples(self) -> int:
        return int(self.hop_s * self.fs)
    @property
    def n_bands(self) -> int:
        return len(self.bands)

def design_filters(cfg: FeatureConfig):
    # 50 Hz notch
    q = 30.0
    b_notch, a_notch = iirnotch(w0=cfg.notch_hz/(cfg.fs/2), Q=q)
    # 0.5–40 Hz band-pass (Butterworth 4th order)
    b_bp, a_bp = butter(4, [cfg.bandpass[0]/(cfg.fs/2), cfg.bandpass[1]/(cfg.fs/2)], btype='bandpass')
    return (b_notch, a_notch), (b_bp, a_bp)

def clean_eeg(window: np.ndarray, cfg: FeatureConfig, filt_coeffs):
    """
    window: (n_channels, window_samples) raw EEG
    Returns filtered EEG of same shape.
    """
    (b_notch, a_notch), (b_bp, a_bp) = filt_coeffs
    x = filtfilt(b_notch, a_notch, window, axis=1)
    x = filtfilt(b_bp, a_bp, x, axis=1)
    return x

def bandpower_welch(x: np.ndarray, fs: int, band: Tuple[float,float], nperseg: Optional[int]=None) -> float:
    """
    Compute bandpower using Welch PSD for a 1-D signal.
    """
    f, Pxx = welch(x, fs=fs, nperseg=nperseg or len(x))
    idx = np.logical_and(f >= band[0], f <= band[1])
    return np.trapz(Pxx[idx], f[idx])

def extract_bandpowers(window: np.ndarray, cfg: FeatureConfig) -> np.ndarray:
    """
    window: (n_channels, window_samples), filtered.
    Returns features: (n_channels, n_bands)
    """
    feats = np.zeros((cfg.n_channels, cfg.n_bands), dtype=np.float32)
    for ch in range(cfg.n_channels):
        sig = window[ch]
        for b_idx, band in enumerate(cfg.bands):
            feats[ch, b_idx] = bandpower_welch(sig, fs=cfg.fs, band=band, nperseg=cfg.welch_nperseg)
    return feats

def stabilise_features(feats: np.ndarray, cfg: FeatureConfig) -> np.ndarray:
    """
    feats: (n_channels, n_bands)
    - relative log-power (optional)
    """
    if cfg.use_relative_log:
        denom = np.sum(feats, axis=1, keepdims=True) + cfg.eps
        feats = np.log((feats + cfg.eps) / denom)
    return feats

class RunningZScore:
    """
    Per-channel running mean/std for baseline normalisation.
    Keeps stats as (n_channels, 1) to apply same scale to all bands of that channel.
    """
    def __init__(self, n_channels: int):
        self.mean = np.zeros((n_channels, 1), dtype=np.float64)
        self.var  = np.ones((n_channels, 1), dtype=np.float64)
        self.n    = np.zeros((n_channels, 1), dtype=np.int64)

    def update(self, feats: np.ndarray):
        # feats: (n_channels, n_bands) -> average over bands to stabilise
        x = feats.mean(axis=1, keepdims=True)  # (n_channels, 1)
        self.n += 1
        delta = x - self.mean
        self.mean += delta / np.maximum(self.n, 1)
        delta2 = x - self.mean
        self.var += delta * delta2

    def apply(self, feats: np.ndarray) -> np.ndarray:
        std = np.sqrt(np.maximum(self.var / np.maximum(self.n - 1, 1), 1e-6))
        return (feats - self.mean) / (std + 1e-6)

# ======== TCN Model ========

class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=5, dilation=1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=0, dilation=dilation)
    def forward(self, x):  # x: (B, C_in, T)
        x = F.pad(x, (self.pad, 0))  # left-pad only -> causal
        return self.conv(x)

class TCNBlock(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size=5, dilation=1, p_drop=0.1):
        super().__init__()
        self.conv = CausalConv1d(ch_in, ch_out, kernel_size, dilation)
        self.norm = nn.LayerNorm(ch_out)  # apply over channel dim, after permute
        self.dropout = nn.Dropout(p_drop)
        self.res = nn.Conv1d(ch_in, ch_out, 1) if ch_in != ch_out else nn.Identity()
    def forward(self, x):  # (B, C_in, T)
        y = self.conv(x)               # (B, C_out, T)
        y = F.relu(y)
        y = y.permute(0, 2, 1)         # (B, T, C_out)
        y = self.norm(y)
        y = y.permute(0, 2, 1)         # back to (B, C_out, T)
        y = self.dropout(y)
        return y + self.res(x)

class EEG_TCN(nn.Module):
    def __init__(self, in_feats=128, hidden=(64,64,64), kernel=5, p_drop=0.1, n_classes=5):
        super().__init__()
        ch = [in_feats] + list(hidden)
        dil = [1,2,4,8][:len(hidden)]
        self.blocks = nn.ModuleList([
            TCNBlock(ch[i], ch[i+1], kernel_size=kernel, dilation=dil[i], p_drop=p_drop)
            for i in range(len(hidden))
        ])
        self.head = nn.Sequential(
            nn.Conv1d(ch[-1], 64, 1),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Conv1d(64, n_classes, 1)
        )
    def forward(self, x):  # x: (B, 128, T)
        for blk in self.blocks:
            x = blk(x)
        logits_seq = self.head(x)  # (B, K, T)
        return logits_seq

# ======== Dataset scaffolding ========

class FeatureSequenceDataset(Dataset):
    """
    Holds sequences of T frames (each frame = 128-dim feature vector) with labels.
    X: (N, T, 128), y: (N,) int labels in [0..K-1]
    """
    def __init__(self, X: np.ndarray, y: np.ndarray):
        assert X.ndim == 3 and X.shape[2] == 128, "Expected X shape (N, T, 128)"
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        x = self.X[idx]             # (T, 128)
        x = torch.from_numpy(x).transpose(0,1)  # -> (128, T) for Conv1d
        y = torch.tensor(self.y[idx])
        return x, y

# ======== Training loop ========

@dataclass
class TrainConfig:
    T: int = 10
    n_classes: int = 5
    batch_size: int = 64
    lr: float = 5e-4
    weight_decay: float = 1e-4
    dropout: float = 0.1
    epochs: int = 30
    kernel: int = 5
    hidden: Tuple[int,...] = (64,64,64)
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_model(train_ds: Dataset, val_ds: Dataset, cfg: TrainConfig):
    model = EEG_TCN(in_feats=128, hidden=cfg.hidden, kernel=cfg.kernel,
                    p_drop=cfg.dropout, n_classes=cfg.n_classes).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False)

    best_val = float('inf'); best_state = None

    for epoch in range(cfg.epochs):
        model.train(); run_loss = 0.0
        for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs} - train"):
            xb = xb.to(cfg.device)  # (B, 128, T)
            yb = yb.to(cfg.device)  # (B,)
            logits_seq = model(xb)          # (B, K, T)
            logits = logits_seq[:, :, -1]   # last time step (causal)
            loss = criterion(logits, yb)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            run_loss += loss.item() * xb.size(0)

        # Validation
        model.eval(); val_loss = 0.0; correct = 0; total = 0
        with torch.no_grad():
            for xb, yb in tqdm(val_loader, desc=f"Epoch {epoch+1}/{cfg.epochs} - val"):
                xb = xb.to(cfg.device); yb = yb.to(cfg.device)
                logits = model(xb)[:, :, -1]
                loss = criterion(logits, yb)
                val_loss += loss.item() * xb.size(0)
                pred = logits.argmax(dim=1)
                correct += (pred == yb).sum().item()
                total += yb.numel()
        val_loss /= max(1, len(val_ds))
        val_acc = correct / max(1, total)
        sched.step()

        print(f"Epoch {epoch+1}: train_loss={(run_loss/len(train_ds)):.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model

# ======== Streaming inference with hysteresis and clutch ========

@dataclass
class HysteresisConfig:
    prob_threshold: float = 0.75
    consecutive_required: int = 3   # N ticks
    rest_class: int = 4             # index of 'rest'

class DecisionSmoother:
    def __init__(self, cfg: HysteresisConfig, n_classes: int):
        self.cfg = cfg
        self.counts = np.zeros(n_classes, dtype=np.int32)
        self.current = cfg.rest_class

    def update(self, probs: np.ndarray) -> int:
        """
        probs: (K,) softmax probabilities at current tick.
        Returns the smoothed class decision.
        """
        k = int(np.argmax(probs))
        if probs[k] >= self.cfg.prob_threshold:
            # increase count for k, decay others
            self.counts *= 0
            self.counts[k] += 1
            if self.counts[k] >= self.cfg.consecutive_required:
                self.current = k
        else:
            # decay counts if not confident
            self.counts *= 0
        return self.current

# ======== Example: building training data from raw EEG ========
# In practice, you'll read your recorded trials here.

def make_feature_frames_from_raw(raw_eeg: np.ndarray, cfg: FeatureConfig,
                                 filt_coeffs, zscaler: Optional[RunningZScore]=None) -> np.ndarray:
    """
    raw_eeg: (n_channels=32, n_samples)
    Returns frames: (T_total, 128) where 128 = 32*4 bands.
    """
    W = cfg.window_samples
    H = cfg.hop_samples
    T_total = max(0, (raw_eeg.shape[1] - W) // H + 1)
    frames = np.zeros((T_total, cfg.n_channels * cfg.n_bands), dtype=np.float32)

    for t in range(T_total):
        start = t*H; stop = start + W
        window = raw_eeg[:, start:stop]
        window = clean_eeg(window, cfg, filt_coeffs)
        bp = extract_bandpowers(window, cfg)
        bp = stabilise_features(bp, cfg)
        if zscaler is not None:
            zscaler.update(bp)
            bp = zscaler.apply(bp)
        frames[t] = bp.reshape(-1)  # (32*4,)
    return frames

# ======== Simple simulator for smoke tests (optional) ========

def simulate_raw_eeg(n_channels=32, n_samples=5000, fs=128, mi='left') -> np.ndarray:
    """
    Generates a crude EEG-like signal with alpha/beta modulation for sanity checks.
    """
    t = np.arange(n_samples) / fs
    x = 5e-6 * np.random.randn(n_channels, n_samples)  # base noise ~ microvolt scale
    # Add alpha (10 Hz) and beta (20 Hz) modulation stronger on "left" channels for 'left' MI.
    left_idx = list(range(0, n_channels//2))
    right_idx = list(range(n_channels//2, n_channels))
    alpha = np.sin(2*np.pi*10*t)
    beta  = np.sin(2*np.pi*20*t)
    if mi == 'left':
        x[left_idx] += (2e-6 * alpha + 1.5e-6 * beta)
    elif mi == 'right':
        x[right_idx] += (2e-6 * alpha + 1.5e-6 * beta)
    return x

# ======== Putting it together (usage examples) ========

if __name__ == "__main__":
    # --- Configs ---
    fcfg = FeatureConfig()
    tcfg = TrainConfig()
    hcfg = HysteresisConfig()

    # --- Filters and baseline normaliser ---
    filt = design_filters(fcfg)
    zscaler = RunningZScore(n_channels=fcfg.n_channels)

    # --- Create toy training/validation sets (replace with real data loader) ---
    # Simulate some raw EEG for left/right/up/grasp/rest
    classes = ['left','right','up','grasp','rest']
    K = len(classes)
    per_class_trials = 50
    seq_T = tcfg.T

    X_seqs = []
    y_seqs = []

    for k, cls in enumerate(classes):
        for _ in range(per_class_trials):
            raw = simulate_raw_eeg(mi=cls if cls in ('left','right') else 'rest',
                                   n_channels=fcfg.n_channels, n_samples=3000, fs=fcfg.fs)
            frames = make_feature_frames_from_raw(raw, fcfg, filt, zscaler=None)
            # Build sequences of length T with label k (toy: label per sequence)
            if frames.shape[0] >= seq_T:
                # Take random slice
                start = np.random.randint(0, frames.shape[0]-seq_T+1)
                seq = frames[start:start+seq_T]  # (T, 128)
                X_seqs.append(seq)
                y_seqs.append(k)

    X = np.stack(X_seqs, axis=0)
    y = np.array(y_seqs, dtype=np.int64)

    # Split train/val
    n = X.shape[0]
    idx = np.random.permutation(n)
    split = int(0.8*n)
    train_ds = FeatureSequenceDataset(X[idx[:split]], y[idx[:split]])
    val_ds   = FeatureSequenceDataset(X[idx[split:]], y[idx[split:]])

    # --- Train ---
    model = train_model(train_ds, val_ds, tcfg)
    model.eval()

    # --- Streaming inference demo (with hysteresis + clutch) ---
    # Simulate a continuous stream and make decisions each hop.
    clutch_enabled = True  # set False to gate outputs
    smoother = DecisionSmoother(hcfg, n_classes=K)

    # Rolling buffer of last T frames
    Tbuf = np.zeros((tcfg.T, 128), dtype=np.float32)

    # Simulated stream: 'left' for a while, then 'rest'
    stream_raw = simulate_raw_eeg(mi='left', n_channels=fcfg.n_channels, n_samples=6000, fs=fcfg.fs)
    frames = make_feature_frames_from_raw(stream_raw, fcfg, filt, zscaler=None)

    with torch.no_grad():
        for t in range(frames.shape[0]):
            # Push new frame
            Tbuf = np.roll(Tbuf, shift=-1, axis=0)
            Tbuf[-1] = frames[t]

            # Only start after buffer filled
            if t < tcfg.T-1:
                continue

            xb = torch.from_numpy(Tbuf).unsqueeze(0).transpose(1,2).to(tcfg.device)  # (1, 128, T)
            logits_seq = model(xb)             # (1, K, T)
            logits = logits_seq[:, :, -1]      # (1, K)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]  # (K,)

            decision = smoother.update(probs)
            decision_name = classes[decision]

            if clutch_enabled and decision_name != 'rest':
                # Send command to robot here (placeholder)
                pass

            if (t % 10) == 0:
                print(f"[tick {t}] probs={np.round(probs, 2)}  -> decision={decision_name}")

    # --- Save model ---
    torch.save(model.state_dict(), "eeg_tcn_demo.pth")
