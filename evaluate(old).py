import torch, numpy as np
from torch.utils.data import DataLoader

from models.st_mamba import NewtonGraphMamba
from datasets.traffic_dataset import TrafficDataset
from utils.metrics import masked_mae, masked_rmse, masked_mape

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = np.load("data/metr_la/metr_la.npz")["data"]
adj = torch.tensor(np.load("data/metr_la/adj.npy")).float().to(device)

test = data[int(.8*len(data)):]
ds = TrafficDataset(test)
loader = DataLoader(ds, batch_size=32)

model = NewtonGraphMamba(in_features=data.shape[-1]).to(device)
model.load_state_dict(torch.load("best_model.pt"))
model.eval()

mae = rmse = mape = 0
with torch.no_grad():
    for X, Y in loader:
        pred = model(X.to(device), adj)
        Y = Y.to(device).permute(0,2,1)
        mae += masked_mae(pred, Y).item()
        rmse += masked_rmse(pred, Y).item()
        mape += masked_mape(pred, Y).item()

print("Test MAE:", mae/len(loader))
print("Test RMSE:", rmse/len(loader))
print("Test MAPE:", mape/len(loader))
