
# train.py
import torch
import numpy as np
from torch.utils.data import DataLoader

from models.st_mamba import NewtonGraphMamba
from datasets.traffic_dataset import TrafficDataset
from utils.metrics import masked_mae, masked_rmse, masked_mape

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    data = np.load("data/metr_la/metr_la.npz")["data"]
    adj = torch.tensor(np.load("data/metr_la/adj.npy")).float().to(device)

    # Split
    T = len(data)
    train_data = data[:int(0.7*T)]
    val_data   = data[int(0.7*T):int(0.8*T)]

    # Dataset
    train_ds = TrafficDataset(train_data)
    val_ds = TrafficDataset(val_data, mean=train_ds.mean, std=train_ds.std)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)

    # Model
    model = NewtonGraphMamba(
        in_features=data.shape[-1],
        num_nodes=data.shape[1]
    ).to(device)

    # Train
    train_model(model, train_loader, val_loader, adj, device)

if __name__ == "__main__":
    main()
