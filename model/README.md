# EEG2Arm — Training & Inference Guide

This README walks you **step‑by‑step** through preparing data, training the model, saving a checkpoint (`.pt`), and running inference. It’s designed to be copy‑paste friendly and beginner‑safe.

---

## What’s in this repo

```
project/
├─ eeg_model.py          # model (CNN stem + Tiny Transformer + Hysteresis etc.)
├─ utils.py              # shared helpers: config, checkpoint save/load, predict
├─ train.py              # CLI to train and save best checkpoint
└─ infer.py              # CLI to run batch inference from a checkpoint
```

---

## Requirements

* Python ≥ 3.9 (3.10+ recommended)
* PyTorch ≥ 2.1 (CUDA optional - Typically neccessary if you don't wanna have stupidly long infer times)
* NumPy
* (Optional) `tqdm` for progress bars

Install:

```bash
pip install torch numpy tqdm
```

---

## Data format (very important)

The training and inference scripts expect NumPy arrays saved as `.npy` files.

* **`X.npy`**: shape **`(N, C, F, T)`**

  * `N` = number of samples (windows)
  * `C` = electrodes (e.g., 32)
  * `F` = bands or feature channels per electrode (e.g., 4 for Δ/α/β/γ). If you only have raw time, you may set `F=1`.
  * `T` = frames/steps per sample (the model pools internally; does **not** have to be fixed if `clip_length=None`).
  * dtype should be `float32` if possible.
* **`y.npy`**: shape **`(N,)`**

  * Integer class indices in `[0, K-1]` where `K` = number of classes you pass to `--classes`.

Quick sanity check in Python:

```python
import numpy as np
X = np.load('X.npy'); y = np.load('y.npy')
print(X.shape, X.dtype)  # (N, C, F, T) float32 preferred
print(y.shape, y.min(), y.max())  # (N,) ints in [0, K-1]
```

> The training script validates that the `C` and `F` observed in your `X.npy` match the `--elec` and `--bands` values you provide.

---

## What gets saved — the checkpoint (`.pt`)

Training writes a **single file** (your chosen path, e.g., `ckpt.pt`) containing:

* `model_state`: the model parameters (weights)
* `config`: the exact model hyperparameters
* `example_input_shape`: a shape like `(1, C, F, T)` used to *materialize* lazy layers when reloading
* `epoch`: epoch number when the checkpoint was saved
* `metrics`: validation metrics at the time of saving (e.g., loss, accuracy)
* `version`: a tag like `eeg_model_v1.5`

This checkpoint is used by `infer.py` to restore the model **exactly** as trained, with all shapes resolved safely.

---

## Quickstart (copy–paste)

1. **Place files** as shown above (`eeg_model.py`, `utils.py`, `train.py`, `infer.py`).

2. **Prepare data** as `X.npy` and `y.npy`.

3. **Train** (saves best model by validation loss):

```bash
python train.py \
  --x X.npy --y y.npy \
  --save ckpt.pt \
  --classes 5 \
  --elec 32 --bands 4 \
  --epochs 30 --batch 64 --lr 3e-4 --wd 1e-4 \
  --val-split 0.2 --normalize
```

4. **Infer** on a batch (or single sample file):

```bash
# X_infer.npy can be (B,C,F,T) or a single (C,F,T)
python infer.py --ckpt ckpt.pt --x X_infer.npy --normalize
```

Expected output:

```
[sample 0] pred=3 conf=0.9123
[sample 1] pred=4 conf=0.7810
...
```

---

## Training — what happens under the hood

* The script loads your data and splits it into **train/val** (default 80/20) with a fixed seed.
* Each batch feeds `(B, C, F, T)` into **EEGModel**:

  1. Per‑electrode depth‑wise 3D convs over `(F, T)`
  2. Early **1×1×1** spatial mix across electrodes
  3. Flatten per frame → **Linear** to `frame_embed`
  4. **Causal Tiny Transformer** over time steps
  5. **Head** → last frame logits `(B, K)`
* Loss: **CrossEntropyLoss** on those logits.
* Optimizer: **AdamW** (with weight decay), mixed precision (AMP), gradient clipping (`1.0`).
* LR schedule: **ReduceLROnPlateau** on validation loss.
* Checkpointing: whenever validation loss improves, the model is saved to `--save`.
* Shapes of lazy layers (e.g., `nn.LazyLinear`) are finalized during the *first* real forward pass.

> Tip: enable `--compile` if you’re on CUDA to try `torch.compile` for speedups.

---

## Inference — what it prints and how to use it

* `infer.py` loads `ckpt.pt`, reconstructs the model, and runs a forward pass on your `X_infer.npy`.
* If you pass a single sample `(C, F, T)`, it is automatically expanded to `(1, C, F, T)`.
* By default it applies `softmax` and prints the **predicted class index** and **confidence** per sample.
* Use `--no-softmax` to print raw **logits** instead of probabilities.
* You can add `--normalize` to apply per‑sample z‑score over `(F, T)` per electrode at inference time (must match whatever you did during training).

Programmatic usage (Python):

```python
from utils import load_model_from_checkpoint
import torch, numpy as np

model, meta = load_model_from_checkpoint('ckpt.pt', device='cuda' if torch.cuda.is_available() else 'cpu')
model.eval()

X = np.load('X_infer.npy')  # (B,C,F,T) or (C,F,T)
if X.ndim == 3:
    X = X[None, ...]
X = torch.from_numpy(X.astype('float32')).to(next(model.parameters()).device)

with torch.no_grad():
    logits = model(X)               # (B, K)
    probs = torch.softmax(logits, -1)
    pred  = probs.argmax(-1)
```

For streaming (frame‑wise) and smoothing, use the model’s `forward_seq` and the `Hysteresis` helper from `eeg_model.py`.

---

## CLI reference — **all options**

### `train.py`

```
usage: train.py [-h] --x X --y Y --save SAVE --classes CLASSES --elec ELEC --bands BANDS
                [--cnn-time-pool CNN_TIME_POOL] [--frame-embed FRAME_EMBED] [--drop DROP]
                [--batch BATCH] [--epochs EPOCHS] [--lr LR] [--wd WD]
                [--val-split VAL_SPLIT] [--seed SEED] [--workers WORKERS]
                [--normalize] [--compile]
```

| Flag              | Required | Type  | Default | Meaning                                             |
| ----------------- | -------- | ----- | ------- | --------------------------------------------------- |
| `--x`             | **Yes**  | path  | –       | Path to `X.npy` with shape `(N,C,F,T)`              |
| `--y`             | **Yes**  | path  | –       | Path to `y.npy` with shape `(N,)` (int labels)      |
| `--save`          | **Yes**  | path  | –       | Output checkpoint path, e.g., `ckpt.pt`             |
| `--classes`       | **Yes**  | int   | –       | Number of classes `K` (labels in `[0, K-1]`)        |
| `--elec`          | **Yes**  | int   | –       | Electrodes `C` expected by the model                |
| `--bands`         | **Yes**  | int   | –       | Bands/features `F` per electrode                    |
| `--cnn-time-pool` | No       | int   | `2`     | Time pooling factor in the first DW block           |
| `--frame-embed`   | No       | int   | `256`   | Transformer embedding size per frame                |
| `--drop`          | No       | float | `0.1`   | Dropout probability throughout the model            |
| `--batch`         | No       | int   | `64`    | Training batch size                                 |
| `--epochs`        | No       | int   | `30`    | Number of epochs                                    |
| `--lr`            | No       | float | `3e-4`  | Learning rate (AdamW)                               |
| `--wd`            | No       | float | `1e-4`  | Weight decay (AdamW)                                |
| `--val-split`     | No       | float | `0.2`   | Fraction of data held out for validation            |
| `--seed`          | No       | int   | `1337`  | RNG seed for reproducibility                        |
| `--workers`       | No       | int   | `2`     | DataLoader workers (increase if I/O bound)          |
| `--normalize`     | No       | flag  | `False` | Apply per‑sample z‑score over `(F,T)` per electrode |
| `--compile`       | No       | flag  | `False` | Try `torch.compile` on CUDA for speed               |

> The script verifies that the `C` and `F` found in `X.npy` match `--elec` and `--bands`.

### `infer.py`

```
usage: infer.py [-h] --ckpt CKPT --x X [--no-softmax] [--normalize]
```

| Flag           | Required | Type | Default | Meaning                                                  |
| -------------- | -------- | ---- | ------- | -------------------------------------------------------- |
| `--ckpt`       | **Yes**  | path | –       | Path to checkpoint produced by training                  |
| `--x`          | **Yes**  | path | –       | Path to `X.npy` `(B,C,F,T)` or single `(C,F,T)`          |
| `--no-softmax` | No       | flag | `False` | Return logits instead of probabilities                   |
| `--normalize`  | No       | flag | `False` | Apply the same per‑sample normalization used in training |

---

## End‑to‑end walkthrough (hand‑holding)

1. **Create a workspace folder** and put your four scripts inside (`eeg_model.py`, `utils.py`, `train.py`, `infer.py`).
2. **Install dependencies** (`pip install torch numpy tqdm`).
3. **Prepare your dataset** as `X.npy` and `y.npy` in the same folder.
4. **Determine values** for `--elec` (= `C`), `--bands` (= `F`), and `--classes` (= number of unique labels).
5. **Run training** using the Quickstart command above. A file like `ckpt.pt` will be created when validation loss improves.
6. **Check console output** for best epoch and validation scores.
7. **Run inference** with `infer.py` on a held‑out `X_infer.npy` to verify predictions.
8. **(Optional) Integrate into your app**: load `ckpt.pt` via `load_model_from_checkpoint` and call `model.forward` or `model.forward_seq` directly.

---

## Troubleshooting & tips

* **Shape mismatch**: “Data shapes (C=…, F=…) do not match config …” → ensure your `--elec` equals the 2nd dim of `X` and `--bands` equals the 3rd dim.
* **Labels out of range**: if any `y` values ≥ `--classes`, fix your labels or update `--classes`.
* **CUDA not used**: install a CUDA build of PyTorch and run on a GPU machine; otherwise the scripts run on CPU.
* **`torch.compile` fails**: it’s experimental—omit `--compile` and train normally.
* **Speed**: increase `--batch` until you run out of memory; tune `--workers`; keep arrays on SSD; consider precomputing features.
* **Normalization**: if you trained with `--normalize`, you should also pass `--normalize` during inference for consistency.
* **Resuming training**: not implemented in `train.py`. You can add a resume option that loads `ckpt.pt`, restores `state_dict`, and continues.

---

## Versioning notes

The checkpoint includes `version: "eeg_model_v1.5"` for basic provenance. If you change model internals or shapes, bump the version string and regenerate checkpoints.

---