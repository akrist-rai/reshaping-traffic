import numpy as np
import torch
from torch.utils.data import Dataset


class TrafficDataset(Dataset):
    """
    Traffic forecasting dataset using sliding windows.

    Input  : past H timesteps  -> [H, N, F]
    Target : next P timesteps  -> [P, N, F]
    """

    def __init__(
        self,
        data: np.ndarray,
        history_len: int = 12,
        horizon: int = 12,
        normalize: bool = True,
        mean: np.ndarray = None,
        std: np.ndarray = None,
        add_time_features: bool = False,
        steps_per_day: int = 288,
    ):
        """
        Parameters
        ----------
        data : np.ndarray
            Traffic data of shape [T, N, F]
        history_len : int
            Number of past timesteps (H)
        horizon : int
            Number of future timesteps to predict (P)
        normalize : bool
            Whether to normalize the data
        mean : np.ndarray
            Precomputed mean for test set normalization
        std : np.ndarray
            Precomputed std for test set normalization
        add_time_features : bool
            Add time-of-day and day-of-week encodings
        steps_per_day : int
            Timesteps per day (e.g. 288 for 5-min data)
        """

        assert data.ndim == 3, "Data must be [T, N, F]"
        self.history_len = history_len
        self.horizon = horizon

        T, N, F = data.shape
        self.num_nodes = N
        self.num_features = F

        # ---------- Time Features ----------
        if add_time_features:
            time_feats = self._build_time_features(T, steps_per_day)
            data = np.concatenate([data, time_feats], axis=-1)
            self.num_features = data.shape[-1]

        # ---------- Normalization (Per Feature) ----------
        if normalize:
            if mean is None:
                self.mean = data.mean(axis=(0, 1), keepdims=True)
                self.std = data.std(axis=(0, 1), keepdims=True) + 1e-6
            else:
                self.mean = mean
                self.std = std

            data = (data - self.mean) / self.std
        else:
            self.mean, self.std = None, None

        self.data = data.astype(np.float32)

        # ---------- Windowing ----------
        self.X, self.Y = self._create_windows()

    def _build_time_features(self, T, steps_per_day):
        """
        Create cyclic time features:
        - sin/cos(hour of day)
        - sin/cos(day of week)
        """
        time = np.arange(T)

        tod = (time % steps_per_day) / steps_per_day
        dow = (time // steps_per_day) % 7 / 7

        tod_sin = np.sin(2 * np.pi * tod)
        tod_cos = np.cos(2 * np.pi * tod)
        dow_sin = np.sin(2 * np.pi * dow)
        dow_cos = np.cos(2 * np.pi * dow)

        time_feats = np.stack(
            [tod_sin, tod_cos, dow_sin, dow_cos], axis=-1
        )  # [T, 4]

        return time_feats[:, None, :]  # [T, 1, 4]

    def _create_windows(self):
        T = self.data.shape[0]
        num_samples = T - self.history_len - self.horizon + 1

        X = np.zeros(
            (num_samples, self.history_len, self.num_nodes, self.num_features),
            dtype=np.float32,
        )
        Y = np.zeros(
            (num_samples, self.horizon, self.num_nodes, self.num_features),
            dtype=np.float32,
        )

        for i in range(num_samples):
            X[i] = self.data[i : i + self.history_len]
            Y[i] = self.data[
                i + self.history_len : i + self.history_len + self.horizon
            ]

        return torch.from_numpy(X), torch.from_numpy(Y)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
