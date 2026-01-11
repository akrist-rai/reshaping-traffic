# utils/metrics.py
import torch

def masked_mae(pred, true, eps=1e-5):
    mask = (true != 0).float()
    loss = torch.abs(pred - true)
    return (loss * mask).sum() / (mask.sum() + eps)

def masked_rmse(pred, true, eps=1e-5):
    mask = (true != 0).float()
    loss = (pred - true) ** 2
    return torch.sqrt((loss * mask).sum() / (mask.sum() + eps))

def masked_mape(pred, true, eps=1e-5):
    mask = (true != 0).float()
    loss = torch.abs((pred - true) / (true + eps))
    return (loss * mask).sum() / (mask.sum() + eps)

