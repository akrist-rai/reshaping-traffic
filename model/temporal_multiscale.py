import torch
import torch.nn as nn
from mamba_ssm import Mamba


class TemporalBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mamba = Mamba(d_model=dim)

    def forward(self, x):
        # x: [B, T, N, F]
        B, T, N, F = x.shape
        x = x.view(B * N, T, F)
        out = self.mamba(x)
        return out[:, -1].view(B, N, F)


class MultiScaleTemporal(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.short = TemporalBlock(dim)  # 5–15 min
        self.mid   = TemporalBlock(dim)  # 30–60 min
        self.long  = TemporalBlock(dim)  # 1–2 hour

        self.attn = nn.Linear(dim, 1)

    def forward(self, x):
        # x: [B, T>=48, N, F]
        short = x[:, -12:]
        mid   = x[:, -24::2]
        long  = x[:, -48::4]

        f_s = self.short(short)
        f_m = self.mid(mid)
        f_l = self.long(long)

        scores = torch.stack([
            self.attn(f_s),
            self.attn(f_m),
            self.attn(f_l)
        ], dim=0)

        weights = torch.softmax(scores, dim=0)
        return weights[0]*f_s + weights[1]*f_m + weights[2]*f_l
#Now your model learns short + long term patterns
