import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from model.temporal_multiscale import MultiScaleTemporal
from model.dynamic_graph import DynamicGraphLearner
from model.uncertainty_head import ProbabilisticHead

# ============================================================
# MAMBA IMPORT (WITH SAFE FALLBACK)
# ============================================================
try:
    from mamba_ssm import Mamba
except ImportError:
    print("WARNING: mamba_ssm not found. Using CPU placeholder.")

    class Mamba(nn.Module):
        def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
            super().__init__()
            self.in_proj = nn.Linear(d_model, d_model * 2)
            self.out_proj = nn.Linear(d_model, d_model)

        def forward(self, x):
            return self.out_proj(F.silu(self.in_proj(x))[:, :, :x.shape[-1]])

# ============================================================
# 1. GAT LAYER (UNCHANGED CORE LOGIC)
# ============================================================
class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, num_heads=4, dropout=0.1):
        super().__init__()
        assert out_features % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = out_features // num_heads

        self.Wq = nn.Linear(in_features, out_features, bias=False)
        self.Wk = nn.Linear(in_features, out_features, bias=False)
        self.Wv = nn.Linear(in_features, out_features, bias=False)

        self.out_proj = nn.Linear(out_features, out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, adj):
        B, N, _ = h.shape

        q = self.Wq(h).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.Wk(h).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.Wv(h).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)

        mask = (adj == 0).view(1, 1, N, N)
        scores = scores.masked_fill(mask, -1e9)

        attn = self.dropout(F.softmax(scores, dim=-1))
        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).reshape(B, N, -1)
        return h + self.out_proj(out)

# ============================================================
# 2. ST-MAMBA BLOCK (SPATIAL + TEMPORAL)
# ============================================================
class STMambaBlock(nn.Module):
    def __init__(self, d_model, num_heads=4, dropout=0.1):
        super().__init__()

        self.gat = GATLayer(d_model, d_model, num_heads, dropout)
        self.norm_s = nn.LayerNorm(d_model)

        self.mamba = Mamba(d_model)
        self.norm_t = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        B, T, N, F = x.shape

        # ---- Spatial ----
        x_s = x.view(B * T, N, F)
        x_s = self.norm_s(x_s + self.dropout(self.gat(x_s, adj)))
        x = x_s.view(B, T, N, F)

        # ---- Temporal ----
        x_t = x.permute(0, 2, 1, 3).reshape(B * N, T, F)
        x_t = self.norm_t(x_t + self.dropout(self.mamba(x_t)))
        x = x_t.view(B, N, T, F).permute(0, 2, 1, 3)

        return x

# ============================================================
# 3. FINAL MODEL (MULTI-SCALE + DYNAMIC GRAPH + UNCERTAINTY)
# ============================================================
class NewtonGraphMamba(nn.Module):
    def __init__(
        self,
        in_features=5,
        d_model=64,
        num_nodes=100,
        num_layers=4,
        num_heads=4,
        prediction_horizon=12
    ):
        super().__init__()

        self.num_nodes = num_nodes

        # ---- Input ----
        self.input_proj = nn.Linear(in_features, d_model)

        # ---- Multi-Scale Temporal Encoder ----
        self.multi_scale_temporal = MultiScaleTemporal(d_model)

        # ---- Dynamic Graph Learner ----
        self.graph_learner = DynamicGraphLearner(num_nodes, d_model)

        # ---- ST Blocks ----
        self.layers = nn.ModuleList([
            STMambaBlock(d_model, num_heads)
            for _ in range(num_layers)
        ])

        # ---- Uncertainty Head ----
        self.head = ProbabilisticHead(d_model)

        self.horizon = prediction_horizon

    def forward(self, x, adj_static):
        """
        x: [B, T>=48, N, F]
        adj_static: [N, N]
        """

        # ---- Input Projection ----
        x = self.input_proj(x)

        # ---- Multi-Scale Temporal Fusion ----
        h = self.multi_scale_temporal(x)  # [B, N, F]

        # ---- Dynamic Graph ----
        adj = self.graph_learner(h, adj_static)

        # ---- Expand temporal dimension for ST blocks ----
        h = h.unsqueeze(1).repeat(1, self.horizon, 1, 1)

        # ---- ST-Mamba Blocks ----
        for layer in self.layers:
            h = layer(h, adj)

        # ---- Prediction + Uncertainty ----
        out = self.head(h[:, -1])  # [B, N, 1]

        return out.squeeze(-1)

# ============================================================
# 4. TEST
# ============================================================
if __name__ == "__main__":
    B, T, N, F = 2, 48, 20, 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.randn(B, T, N, F).to(device)
    adj = torch.eye(N).to(device)

    model = NewtonGraphMamba(
        in_features=F,
        d_model=32,
        num_nodes=N,
        num_layers=2,
        prediction_horizon=6
    ).to(device)

    y = model(x, adj)
    print("✅ Forward pass OK")
    print("Output shape:", y.shape)
