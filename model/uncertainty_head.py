import torch
import torch.nn as nn


class ProbabilisticHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        return self.fc(self.dropout(x))


def mc_dropout_predict(model, x, runs=30):
    model.train()
    preds = torch.stack([model(x) for _ in range(runs)])
    return preds.mean(0), preds.std(0)

# Gives mean + uncertainty
# Essential for traffic routing safety
