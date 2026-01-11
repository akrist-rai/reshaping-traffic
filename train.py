import torch, numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models.st_mamba import NewtonGraphMamba
from datasets.traffic_dataset import TrafficDataset
from utils.metrics import masked_mae
from utils.seed import set_seed

def main():
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = np.load("data/metr_la/metr_la.npz")["data"]
    adj = torch.tensor(np.load("data/metr_la/adj.npy")).float().to(device)

    T = len(data)
    train, val = data[:int(.7*T)], data[int(.7*T):int(.8*T)]

    train_ds = TrafficDataset(train)
    val_ds = TrafficDataset(val, mean=train_ds.mean, std=train_ds.std)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)

    model = NewtonGraphMamba(
        in_features=data.shape[-1]
    ).to(device)

    opt = AdamW(model.parameters(), lr=1e-3)
    sched = ReduceLROnPlateau(opt, patience=5)

    best = 1e9
    for epoch in range(50):
        model.train()
        for X, Y in train_loader:
            X, Y = X.to(device), Y.to(device).permute(0,2,1)
            opt.zero_grad()
            loss = masked_mae(model(X, adj), Y)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = sum(
                masked_mae(model(X.to(device), adj),
                           Y.to(device).permute(0,2,1)).item()
                for X, Y in val_loader
            ) / len(val_loader)

        sched.step(val_loss)
        if val_loss < best:
            best = val_loss
            torch.save(model.state_dict(), "best_model.pt")

        print(f"Epoch {epoch:03d} | Val MAE {val_loss:.4f}")

if __name__ == "__main__":
    main()
