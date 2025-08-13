#!/usr/bin/env python3
# ===============================================================
# v1.0
# EEG-to-Robot-Arm — Inference script
# - Loads checkpoint (*.pt)
# - Runs inference on X.npy ((B,C,F,T) or single (C,F,T))
# ===============================================================

import argparse
from utils import predict_npy  # shared prediction

def build_argparser():
    p = argparse.ArgumentParser(description="EEG2Arm — run inference from checkpoint")
    p.add_argument("--ckpt", required=True, help="Path to checkpoint *.pt")
    p.add_argument("--x", required=True, help="Path to X.npy (B,C,F,T) or (C,F,T)")
    p.add_argument("--no-softmax", action="store_true", help="Return logits, not probabilities")
    p.add_argument("--normalize", action="store_true", help="Per-sample z-score over (F,T)")
    return p

def main():
    ap = build_argparser()
    args = ap.parse_args()

    idx, conf, raw = predict_npy(
        ckpt_path=args.ckpt,
        X_path=args.x,
        apply_softmax=not args.no_softmax,
        per_sample_normalize=args.normalize,
    )

    # Pretty print a small summary
    for i, (c, p) in enumerate(zip(idx, conf)):
        print(f"[sample {i}] pred={int(c)} conf={float(p):.4f}")

if __name__ == "__main__":
    main()