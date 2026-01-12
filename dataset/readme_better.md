
```md
# TrafficDataset – Spatio-Temporal Traffic Forecasting

This repository provides a **research-grade PyTorch Dataset** for traffic forecasting using sliding windows.  
It is compatible with **STGCN, DCRNN, ST-Mamba, Graph Transformers**, and other spatio-temporal models.

---

## 📊 Data Format

The dataset expects traffic data in the following format:

```

data: np.ndarray [T, N, F]

````

| Symbol | Meaning |
|------|--------|
| `T` | Total timesteps |
| `N` | Number of traffic sensors / nodes |
| `F` | Number of features per node |

### Example
```python
data.shape = (28800, 207, 3)
````

| Feature Index | Meaning   |
| ------------- | --------- |
| 0             | Speed     |
| 1             | Flow      |
| 2             | Occupancy |

---

## 🧠 Problem Formulation

The dataset converts time-series data into supervised samples:

```
Past H steps ─────────▶ Predict next P steps
[H, N, F]                [P, N, F]
```

---

## ⚙️ Parameters

```python
TrafficDataset(
    data,
    history_len=12,
    horizon=12,
    normalize=True,
    add_time_features=True
)
```

| Parameter           | Description                  |
| ------------------- | ---------------------------- |
| `history_len (H)`   | Past timesteps used as input |
| `horizon (P)`       | Future timesteps to predict  |
| `normalize`         | Per-feature normalization    |
| `add_time_features` | Adds temporal encodings      |

---

## ⏱️ Time Features (Optional)

When enabled, the dataset adds **4 cyclic features**:

* sin(hour of day)
* cos(hour of day)
* sin(day of week)
* cos(day of week)

This significantly improves forecasting accuracy.

---

## 🔄 Normalization

Normalization is done **per feature across time and nodes**:

```text
mean = mean over (T, N)
std  = std over (T, N)
```

This matches best practices used in:

* METR-LA
* PeMS-BAY
* DCRNN
* STGCN

---

## 📐 Output Shapes

After windowing:

| Tensor | Shape                    |
| ------ | ------------------------ |
| `X`    | `[num_samples, H, N, F]` |
| `Y`    | `[num_samples, P, N, F]` |

Batch shapes from DataLoader:

```text
X → [B, H, N, F]
Y → [B, P, N, F]
```

---

## 🚀 Usage Example

```python
from torch.utils.data import DataLoader
from datasets.traffic_dataset import TrafficDataset

dataset = TrafficDataset(
    data,
    history_len=12,
    horizon=12,
    add_time_features=True
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

---

## ✅ Features

✔ Sliding-window forecasting
✔ Multi-feature prediction
✔ Time-of-day & day-of-week encoding
✔ Per-feature normalization
✔ Research benchmark compatible
✔ Efficient NumPy → Torch pipeline

---

## 🔌 Compatible Models

* STGCN
* DCRNN
* ST-Mamba
* Temporal CNNs
* Graph Transformers

---

## 📌 Summary

This dataset follows **industry and academic best practices** for traffic forecasting and can be directly plugged into modern spatio-temporal deep learning models.

---


