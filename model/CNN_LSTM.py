#!/usr/bin/env python3
# ===============================================================
#  EEG-to-Robot-Arm (v1.1)
# ===============================================================

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ----------------------------------------------------------------
# 0.  Utility: sinusoidal position encoding  (For Transformer positioning)
# ----------------------------------------------------------------
class PositionalEncoding(nn.Module):
    """
    Need to double check pulled from online resource
    """
    def __init__(self, d_model: int, max_len: int = 10_000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float)
                             * (-math.log(10_000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))         # (1, max_len, d)

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, T, d).  Adds sin-cos positions for first T steps."""
        return x + self.pe[:, :x.size(1)]


# ----------------------------------------------------------------
# 1.  Graph utilities
# ----------------------------------------------------------------
def make_electrode_adj(n_elec: int,
                       edges: list[tuple[int, int]],
                       device: torch.device,
                       self_loop: bool = True) -> Tensor:
    """
    Fixed adjacency

    n_elec   - number of electrodes (nodes)  
    edges    - undirected pairs (i,j).  Put 10-20 per node for good coverage.
    """
    A = torch.zeros(n_elec, n_elec, device=device)
    for i, j in edges:
        if i >= n_elec or j >= n_elec:
            raise ValueError(f"edge ({i},{j}) exceeds n_elec={n_elec}")
        A[i, j] = A[j, i] = 1.0
    if self_loop:
        A.fill_diagonal_(1.0)

    D_inv_sqrt = torch.diag(1.0 / torch.sqrt(A.sum(1)).clamp(min=1e-6))
    return D_inv_sqrt @ A @ D_inv_sqrt           


# ----------------------------------------------------------------
# 2.  Building blocks
# ----------------------------------------------------------------
class DWConvBlock(nn.Module):
    """Depth-wise 3-D conv → BN → ReLU → optional pool."""
    def __init__(self, in_ch: int, k_freq: int = 3,
                 k_time: int = 3, pool: tuple[int, int, int] | None = (1, 2, 2)):
        super().__init__()
        self.dw = nn.Conv3d(in_ch, in_ch, (1, k_freq, k_time),
                            padding=(0, k_freq // 2, k_time // 2),
                            groups=in_ch, bias=False)
        self.bn = nn.BatchNorm3d(in_ch)
        self.pool = nn.MaxPool3d(pool) if pool else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = self.dw(x)
        x = self.bn(x)
        x = F.relu(x, inplace=True)
        return self.pool(x)


class PointwiseMix(nn.Module):
    """1*1*1 conv that can mix electrodes (groups=1) or keep them isolated."""
    def __init__(self, in_ch: int, out_ch: int, groups: int = 1):
        super().__init__()
        self.pw = nn.Conv3d(in_ch, out_ch, 1, groups=groups, bias=False)
        self.bn = nn.BatchNorm3d(out_ch)

    def forward(self, x: Tensor) -> Tensor:
        return F.relu(self.bn(self.pw(x)), inplace=True)


class GCNLayer(nn.Module):
    """Residual Kipf-Welling layer with (optional) edge dropout."""
    def __init__(self, in_f: int, out_f: int, adj: Tensor, p_drop_edge: float = 0.1):
        super().__init__()
        if adj.shape[0] != adj.shape[1]:
            raise ValueError("Adjacency must be square")
        self.register_buffer("A_hat", adj)           # fixed, non-trainable
        self.W = nn.Linear(in_f, out_f, bias=False)
        self.bn = nn.BatchNorm1d(out_f)
        self.p_drop_edge = p_drop_edge

        # To allow residual even when in_f ≠ out_f
        self.res_proj = nn.Identity() if in_f == out_f else nn.Linear(in_f, out_f)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B*frames, N, F)
        A = self.A_hat
        if self.training and self.p_drop_edge > 0:
            mask = torch.rand_like(A) > self.p_drop_edge
            A = A * mask

        h = self.W(x)                                # (B*frames, N, out_f)
        h = torch.bmm(A.expand(h.size(0), *A.shape), h)
        h = self.bn(h.transpose(1, 2)).transpose(1, 2)
        return F.relu(h + self.res_proj(x), inplace=True)


class TinyTransformer(nn.Module):
    """
    2-layer causal Transformer on the frame axis.
    Uses *sinusoidal* positional encoding, so no re-init needed for longer clips.
    """
    def __init__(self, embed: int = 1024, n_heads: int = 4, ff: int = 2048):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(embed, n_heads, ff,
                                               dropout=0.1, batch_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, 2)
        self.pos = PositionalEncoding(embed)

    def forward(self, x: Tensor) -> Tensor:
        """
        x : (B, T, embed). Returns last-time-step ctx (B, embed)
        """
        B, T, E = x.shape
        causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        z = self.enc(self.pos(x), causal_mask)
        return z[:, -1]                              # (B, E)


# ----------------------------------------------------------------
# 3.  Complete model
# ----------------------------------------------------------------
class EEG2Arm(nn.Module):
    """
    Args
    ----
    n_elec            : number of electrodes (nodes)
    n_bands           : spectral bins (frequency axis before pooling)
    cnn_time_pool     : factor the first DW block pools along time
    n_classes         : output dim (logits or regression targets)
    pointwise_groups  : 1 → mix electrodes early, n_elec → keep separate
    """
    def __init__(self,
                 n_elec: int = 32,
                 n_bands: int = 5,
                 clip_length: int | None = None,      # only used for sanity asserts
                 cnn_time_pool: int = 2,
                 n_classes: int = 5,
                 pointwise_groups: int = 1,
                 edges: list[tuple[int, int]] | None = None):
        super().__init__()
        self.n_elec = n_elec
        self.clip_length = clip_length

        # ------------- 3-D CNN stem -------------
        self.dw1 = DWConvBlock(n_elec, pool=(1, 2, cnn_time_pool))
        self.dw2 = DWConvBlock(n_elec, pool=(1, 1, 1))
        self.mix = PointwiseMix(n_elec, 2 * n_elec, groups=pointwise_groups)

        # Shapes after the two DW blocks
        #  F' = ceil(n_bands / 2) due to pool(2)   (roughly)
        #  T' = ceil(clip_length / cnn_time_pool)
        self.register_buffer("dummy_adj", torch.tensor(0.))  # placeholder

        # ------------- Graph layers -------------
        adj = make_electrode_adj(n_elec,
                                 edges or [],          # empty = I_N
                                 device=torch.device("cpu"))
        self.gcn1 = GCNLayer(in_f=6,  out_f=32, adj=adj, p_drop_edge=0.1)
        self.gcn2 = GCNLayer(in_f=32, out_f=32, adj=adj, p_drop_edge=0.1)

        # ------------- Transformer -------------
        self.tform = TinyTransformer(embed=32 * n_elec)

        # ------------- Head -------------
        self.fc = nn.Sequential(
            nn.Linear(32 * n_elec, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes)
        )

    # ------------------------------------------------------------
    #  Forward
    # ------------------------------------------------------------
    def forward(self, x: Tensor) -> Tensor:
        """
        x : (B, n_elec, n_bands, T)
        """
        B, C, F0, T0 = x.shape
        assert C == self.n_elec, f"expected {self.n_elec} electrodes, got {C}"
        if self.clip_length is not None:
            assert T0 == self.clip_length, f"expected {self.clip_length} frames, got {T0}"

        # 1. make 5-D (Conv3d wants D)
        x = x.unsqueeze(2)                           # (B, C, 1, F0, T0)

        # 2. depth-wise CNN
        x = self.dw1(x)                              # (B, C, 1, F1, T1)
        x = self.dw2(x)                              # (B, C, 1, F2, T2)
        x = self.mix(x)                              # (B, 2C, 1, F2, T2)
        x = x.squeeze(2)                             # (B, 2C, F2, T2)

        # 3. reshape for graph
        B, ch, Fp, Tp = x.shape                      # ch = 2C
        k = ch // self.n_elec                        # k=2
        x = x.view(B, self.n_elec, k, Fp, Tp)        # (B, C, 2, Fp, Tp)
        x = x.permute(0, 4, 1, 2, 3)                 # (B, Tp, C, 2, Fp)
        x = x.reshape(B * Tp, self.n_elec, k * Fp)   # (B*Tp, C, 2*Fp)

        # 4. graph conv
        x = self.gcn1(x)                             # (B*Tp, C, 32)
        x = self.gcn2(x)                             # (B*Tp, C, 32)

        # 5. flatten to frame embeddings
        x = x.reshape(B, Tp, -1)                     # (B, Tp, 32*C)

        # 6. Transformer over frames
        ctx = self.tform(x)                          # (B, 32*C)

        # 7. MLP head
        return self.fc(ctx)


# ----------------------------------------------------------------
# 4.  Smoke test
# ----------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    dummy_frames = 12                   # try any length
    dummy = torch.randn(4, 32, 5, dummy_frames)

    # simple 10-neighbour ring as example edges
    ring_edges = [(i, (i + 1) % 32) for i in range(32)]
    model = EEG2Arm(n_elec=32,
                    n_bands=5,
                    clip_length=None,        # accept any T
                    cnn_time_pool=2,
                    n_classes=5,
                    pointwise_groups=1,
                    edges=ring_edges).to(dummy.device)

    out = model(dummy)
    print("out shape:", out.shape)           # (4, 5)


# 

# pointwise_groups – letting electrodes mix earlier can help.
# Transformer depth / head count – two layers is tiny for long clips.
# cnn_time_pool – bigger pool compresses the frame axis faster, making the Transformer cheaper.