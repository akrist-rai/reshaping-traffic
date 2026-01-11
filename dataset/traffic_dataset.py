
# datasets/traffic_dataset.py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class TrafficDataset(Dataset):
    def __init__(self, data, history_len=12, horizon=12, mean=None, std=None):
        """
        data: np.ndarray [T, N, F]
        """
        self.history_len = history_len
        self.horizon = horizon
        
        if mean is None:
            self.mean = data.mean()
            self.std = data.std() + 1e-6
        else:
            self.mean = mean
            self.std = std
        
        # Normalize
        self.data = (data - self.mean) / self.std
        
        self.X, self.Y = self._create_windows()

    def _create_windows(self):
        X, Y = [], []
        T = self.data.shape[0]
        
        for t in range(T - self.history_len - self.horizon):
            X.append(self.data[t:t+self.history_len])
            Y.append(self.data[t+self.history_len:t+self.history_len+self.horizon, :, 0])
            # target = first feature (speed)
        
        return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

