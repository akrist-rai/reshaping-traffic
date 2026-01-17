import torch
import torch.nn as nn


class DynamicGraphLearner(nn.Module):
    def __init__(self, num_nodes, dim):
        super().__init__()
        self.node_emb = nn.Parameter(torch.randn(num_nodes, dim))
        self.proj = nn.Linear(dim, dim)

    def forward(self, h, A_static):
        # h: [B, N, F]
        q = self.proj(h)
        k = self.node_emb.unsqueeze(0)

        A_dyn = torch.softmax(
            torch.matmul(q, k.transpose(-1, -2)), dim=-1
        )

        return 0.7 * A_static + 0.3 * A_dyn

#Graph now adapts to congestion, events, time
