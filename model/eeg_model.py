#!/usr/bin/env python3
# ===============================================================
#  EEG-to-Robot-Arm model (tiny demo version)
#  -- Depth-wise 3-D CNN  →  Graph Conv  →  Tiny Transformer head
# ===============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ------------------------------------------------------------
# 1.  Graph utilities: fixed 10–20 adjacency  (32 × 32 matrix)
# ------------------------------------------------------------
def make_electrode_adj(device: torch.device) -> Tensor:
    """
    Return the (normalised) adjacency matrix Ũ for 32 EEG electrodes.
    NOTE: Fill `edges` with the undirected pairs you actually want.
    """
    edges: list[tuple[int, int]] = [
        # e.g. (0, 1), (1, 2), ...
    ]

    A = torch.zeros(32, 32, device=device)
    for i, j in edges:
        A[i, j] = A[j, i] = 1.0               # undirected edge
    A.fill_diagonal_(1.0)                    # self-loops

    D_inv_sqrt = torch.diag(1.0 / torch.sqrt(A.sum(1)))
    return D_inv_sqrt @ A @ D_inv_sqrt       # Ũ = D⁻¹⸍² A D⁻¹⸍²


# ----------------------------------------------------------------
# 2.  Basic building blocks
# ----------------------------------------------------------------
class DWConvBlock(nn.Module):
    """Depth-wise 3-D conv → BN → ReLU → optional pool."""
    def __init__(self, in_ch: int, k_freq: int = 3,
                 k_time: int = 3, pool: tuple | None = (1, 2, 2)):
        super().__init__()
        self.dw = nn.Conv3d(in_ch, in_ch, (1, k_freq, k_time),
                            padding=(0, k_freq // 2, k_time // 2),
                            groups=in_ch, bias=False)
        self.bn = nn.BatchNorm3d(in_ch)
        self.pool = nn.MaxPool3d(pool) if pool else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.pool(F.relu(self.bn(self.dw(x))))


class PointwiseMix(nn.Module):
    """1×1×1 conv that can mix electrodes (groups=1) or keep them isolated."""
    def __init__(self, in_ch: int, out_ch: int, groups: int = 1):
        super().__init__()
        self.pw = nn.Conv3d(in_ch, out_ch, 1, groups=groups, bias=False)
        self.bn = nn.BatchNorm3d(out_ch)

    def forward(self, x: Tensor) -> Tensor:
        return F.relu(self.bn(self.pw(x)))


class GCNLayer(nn.Module):
    """Frame-wise Kipf-Welling GCN layer."""
    def __init__(self, in_f: int, out_f: int, adj: Tensor):
        super().__init__()
        # Make sure the adjacency sits on the same device as W
        self.register_buffer("A_hat", adj)
        self.W = nn.Linear(in_f, out_f, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B*T', nodes=32, in_f)
        h = self.W(x)                                            # linear
        a = self.A_hat.expand(h.size(0), *self.A_hat.shape)      # (B*T',32,32)
        h = torch.bmm(a, h)                                      # graph conv
        return F.relu(h)


class TinyTransformer(nn.Module):
    """2-layer causal Transformer over frame axis."""
    def __init__(self, embed: int = 128, n_heads: int = 4, ff: int = 256):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(embed, n_heads, ff,
                                               dropout=0.1, batch_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, 2)
        self.pos = nn.Parameter(torch.randn(1, 10, embed))   # supports ≤10 frames

    def forward(self, x: Tensor) -> Tensor:
        """
        Args
        ----
        x : (B, T, embed)  – sequence of frame embeddings
        Returns
        -------
        (B, embed) – embedding of the **last** frame (causal context)
        """
        B, T, _ = x.shape
        # causal mask (upper-triangular, excluding main diag)
        mask = torch.triu(torch.ones(T, T, device=x.device), 1).bool()
        z = self.enc(x + self.pos[:, :T], mask)
        return z[:, -1]                                     # final step


# ----------------------------------------------------------------
# 3.  Complete model
# ----------------------------------------------------------------
class EEG2Arm(nn.Module):
    def __init__(self, n_classes: int = 5, pointwise_groups: int = 1):
        super().__init__()

        # 3-D DW-CNN stem
        self.dw1 = DWConvBlock(32)               # keeps (32, F=5, T=10) → (32, 3, 5)
        self.dw2 = DWConvBlock(32, pool=(1, 1, 1))
        self.mix = PointwiseMix(32, 64, groups=pointwise_groups)

        # Graph layers
        adj = make_electrode_adj(torch.device("cpu"))
        self.gcn1 = GCNLayer(in_f=6,  out_f=32, adj=adj)    # 6 = (64//32)*3
        self.gcn2 = GCNLayer(in_f=32, out_f=32, adj=adj)

        # Tiny Transformer
        self.tform = TinyTransformer(embed=32 * 32)         # 1024-dim frames

        # Classification / regression head
        self.fc = nn.Sequential(
            nn.Linear(32 * 32, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes)   # change n_classes + loss for regression
        )

    # ------------------------------------------------------------
    #  Forward
    # ------------------------------------------------------------
    def forward(self, x: Tensor) -> Tensor:
        """
        Args
        ----
        x : (B, C=32, F=5, T=10)
        """
        # 1. Add dummy depth dim expected by Conv3d
        x = x.unsqueeze(2)                           # (B,32,1,5,10)

        # 2. Depth-wise CNN
        x = self.dw1(x)                              # (B,32,1,3,5)
        x = self.dw2(x)                              # (B,32,1,3,5)
        x = self.mix(x)                              # (B,64,1,3,5)
        x = x.squeeze(2)                             # (B,64,3,5)

        # 3. Reshape to (B*T', nodes, feats)
        B, ch, Fp, Tp = x.shape                      # ch=64, Fp=3, Tp=5
        k = ch // 32                                 # k=2
        x = x.view(B, 32, k, Fp, Tp)                 # (B,32,2,3,5)
        x = x.permute(0, 4, 1, 2, 3)                 # (B,5,32,2,3)
        x = x.reshape(B * Tp, 32, k * Fp)            # (B*5,32,6)

        # 4. Graph convolution
        x = self.gcn1(x)                             # (B*5,32,32)
        x = self.gcn2(x)                             # (B*5,32,32)

        # 5. Flatten node×feat → frame embeddings
        x = x.reshape(B, Tp, -1)                     # (B,5,1024)

        # 6. Tiny Transformer (causal)
        ctx = self.tform(x)                          # (B,1024)

        # 7. Head
        return self.fc(ctx)                          # logits / velocities


# ----------------------------------------------------------------
# 4.  Sanity check
# ----------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    # fake batch: 8 samples, 32 electrodes, 5 spectral bands, 10 frames
    dummy = torch.randn(8, 32, 5, 10)

    model = EEG2Arm(n_classes=5, pointwise_groups=1)      # groups=32 isolates electrodes
    out = model(dummy)

    print("Output:", out.shape)       # → torch.Size([8, 5])
