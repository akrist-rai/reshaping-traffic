

# TrafficDataset – Spatio-Temporal Traffic Forecasting Dataset

This module defines a PyTorch `Dataset` for **traffic speed forecasting** using a sliding-window approach.
It is compatible with most **spatio-temporal deep learning models** such as STGCN, DCRNN, ST-Mamba, and temporal transformers.

---

## 📁 File Structure

```text
datasets/
└── traffic_dataset.py
```

---

## 📌 Overview

The dataset converts raw traffic time-series data into supervised learning samples:

* **Input (X)** → past `history_len` timesteps
* **Target (Y)** → future `horizon` timesteps

This is the standard formulation used in traffic forecasting benchmarks like **METR-LA** and **PeMS**.

---

## 📊 Expected Data Format

```python
data: np.ndarray [T, N, F]
```

| Symbol | Meaning                                                    |
| ------ | ---------------------------------------------------------- |
| `T`    | Total number of timesteps                                  |
| `N`    | Number of sensors / nodes                                  |
| `F`    | Number of features per node (e.g., speed, flow, occupancy) |

Example:

```python
data.shape = (28800, 207, 3)
```

---

## ⚙️ Dataset Parameters

```python
TrafficDataset(
    data,
    history_len=12,
    horizon=12,
    mean=None,
    std=None
)
```

| Parameter     | Description                                            |
| ------------- | ------------------------------------------------------ |
| `history_len` | Number of past timesteps used as input                 |
| `horizon`     | Number of future timesteps to predict                  |
| `mean`        | Optional precomputed mean (for test set normalization) |
| `std`         | Optional precomputed std                               |

---

## 🔄 Normalization

* If `mean` and `std` are not provided, **global normalization** is applied:

```python
(data - mean) / std
```

* A small epsilon (`1e-6`) is added for numerical stability.

> ℹ️ This ensures stable and faster training.

---

## 🧠 Sliding Window Construction

For each time index `t`:

### Input (X)

```text
[t : t + history_len] → [H, N, F]
```

### Target (Y)

```text
[t + history_len : t + history_len + horizon] → [P, N]
```

* Only **feature index 0** is predicted (typically traffic speed).

---

## 📐 Tensor Shapes

After preprocessing:

```text
X.shape = [num_samples, history_len, N, F]
Y.shape = [num_samples, horizon, N]
```

Each dataset item returns:

```python
X, Y
```

| Tensor | Shape       |
| ------ | ----------- |
| `X`    | `[H, N, F]` |
| `Y`    | `[P, N]`    |

---

## 🚀 Usage Example

```python
from torch.utils.data import DataLoader
from datasets.traffic_dataset import TrafficDataset

dataset = TrafficDataset(
    data,
    history_len=12,
    horizon=12
)

loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    drop_last=True
)

for X, Y in loader:
    print(X.shape, Y.shape)
```

Batch shapes:

```text
X → [B, H, N, F]
Y → [B, P, N]
```

---

## ✅ Design Highlights

✔ Research-grade sliding window logic
✔ Compatible with graph-based models
✔ Supports multi-step forecasting
✔ Torch-ready tensors
✔ Clean and modular implementation

---

## 🔧 Possible Improvements

### 1️⃣ Per-Feature Normalization (Recommended)

Instead of global normalization:

```python
mean = data.mean(axis=(0,1), keepdims=True)
std  = data.std(axis=(0,1), keepdims=True)
```

---

### 2️⃣ Multi-Feature Prediction

Predict all features instead of only speed:

```python
Y.append(...[:, :, :])
```

Output:

```text
Y → [horizon, N, F]
```

---

### 3️⃣ Time Encoding (Advanced)

Add:

* Hour of day
* Day of week

This significantly improves performance in traffic forecasting models.

---

## 🧪 Compatible Models

This dataset works seamlessly with:

* STGCN
* DCRNN
* ST-Mamba
* Temporal CNNs
* Graph Transformers

---

## 📌 Summary

This dataset follows **standard practices used in real traffic forecasting research** and is suitable for both academic and production-level experimentation.

