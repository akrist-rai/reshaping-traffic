import torch
import numpy as np
from torch.utils.data import DataLoader

from models.st_mamba import NewtonGraphMamba
from datasets.traffic_dataset import TrafficDataset
from utils.metrics import masked_mae, masked_rmse, masked_mape

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =======================
# LOAD DATA
# =======================
data = np.load("data/metr_la/metr_la.npz")["data"]
adj = torch.tensor(np.load("data/metr_la/adj.npy")).float().to(device)

test = data[int(.8 * len(data)):]
ds = TrafficDataset(test)

loader = DataLoader(
    ds,
    batch_size=4,  # small batch for safety
    pin_memory=True
)

# =======================
# MODEL
# =======================
model = NewtonGraphMamba(
    in_features=data.shape[-1]
).to(device)

model.load_state_dict(torch.load("best_model.pt", map_location=device))
model.eval()

mae = rmse = mape = 0.0

# =======================
# EVALUATION
# =======================
with torch.no_grad():
    for X, Y in loader:
        X = X.to(device, non_blocking=True)
        Y = Y.to(device, non_blocking=True).permute(0, 2, 1)

        pred = model(X, adj)

        mae += masked_mae(pred, Y).item()
        rmse += masked_rmse(pred, Y).item()
        mape += masked_mape(pred, Y).item()

mae /= len(loader)
rmse /= len(loader)
mape /= len(loader)

print("Test MAE :", mae)
print("Test RMSE:", rmse)
print("Test MAPE:", mape)
