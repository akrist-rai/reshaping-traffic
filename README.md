

# 🚦 reshape-traffic: Spatio-Temporal Traffic Forecasting

A **Spatio-Temporal Deep Learning model** for traffic forecasting that combines
**Graph Attention Networks (GAT)** for spatial dependency modeling and **Mamba (SSM)** for long-range temporal modeling.

This project is implemented in **PyTorch** and evaluated on the **METR-LA traffic dataset**.

---

## 📌 Key Features

* ✅ **Graph Attention Networks (GAT)** for spatial relationships between sensors
* ✅ **Mamba (State Space Model)** for efficient long-range temporal modeling
* ✅ Sliding-window **time series forecasting**
* ✅ Masked metrics for missing traffic data
* ✅ Clean training & evaluation pipeline
* ✅ GPU-ready (CUDA supported)

---

## 🧠 Model Architecture

```
Input Traffic Data
   ↓
Input Projection (Linear)
   ↓
[ ST-Mamba Block × N ]
   ├─ Spatial Modeling (Multi-Head GAT)
   ├─ Temporal Modeling (Mamba SSM)
   └─ Residual + LayerNorm
   ↓
Prediction Head
   ↓
Traffic Forecast (Next H time steps)
```

### Why this design?

* **GAT** captures *road network topology*
* **Mamba** handles *long temporal dependencies* efficiently
* Interleaving both gives strong **spatio-temporal reasoning**

---

## 📂 Project Structure

```
.
├── models/
│   └── st_mamba.py          # NewtonGraphMamba model
├── datasets/
│   └── traffic_dataset.py  # Sliding window dataset + normalization
├── utils/
│   ├── metrics.py          # Masked MAE / RMSE / MAPE
│   └── seed.py             # Reproducibility
├── data/
│   └── metr_la/
│       ├── metr_la.npz     # Traffic data
│       └── adj.npy         # Adjacency matrix
├── train.py                # Training loop
├── evaluate.py             # Model evaluation
├── best_model.pt           # Saved checkpoint
└── README.md
```

---

## 📊 Dataset

**METR-LA Traffic Dataset**

* Traffic speed readings from **207 sensors**
* Collected every **5 minutes**
* Graph adjacency based on road distances

**Data shape**

```
[T, N, F]
T = time steps
N = number of sensors
F = features (speed, etc.)
```

---

## 🏗 Dataset Pipeline

* Sliding window approach:

  * **History length** = 12
  * **Prediction horizon** = 12
* Z-score normalization using **training statistics**
* Target = **traffic speed (feature index 0)**

---

## ⚙️ Training

### Run training

```bash
python train.py
```

### Training details

* Optimizer: **AdamW**
* Loss: **Masked MAE**
* LR Scheduler: **ReduceLROnPlateau**
* Checkpoint: saves **best validation MAE**

Output:

```
Epoch 012 | Val MAE 2.3471
```

---

## 📈 Evaluation

### Run evaluation

```bash
python evaluate.py
```

### Metrics

* **MAE** – Mean Absolute Error
* **RMSE** – Root Mean Squared Error
* **MAPE** – Mean Absolute Percentage Error

Masked metrics ensure missing values do not affect results.

---

## 🧪 Model Input & Output

**Input**

```
[Batch, History, Nodes, Features]
```

**Output**

```
[Batch, Nodes, Prediction_Horizon]
```

---

## 🖥 Hardware Support

* ✅ CPU
* ✅ CUDA GPU
* ⚠️ `mamba_ssm` required for full performance

  * Fallback CPU implementation included for testing

---

## 📦 Dependencies

```txt
torch
numpy
mamba_ssm
```

Install:

```bash
pip install torch numpy mamba-ssm
```

---


## 🚦 Phase-2: Traffic Routing & Flow Control (Closed-Loop)

This project extends beyond forecasting into active traffic control.
System Overview

Current Traffic
     ↓
Model-1 (NewtonGraphMamba)
     ↓
Future Traffic Prediction
     ↓
Model-2 (Routing Controller)
     ↓
Route Allocation (75 / 25)
     ↓
Traffic Flow Change
     ↓
Feedback → Next Prediction

Key Design Decisions

    Separation of prediction and control

    Congestion-aware routing (not shortest-path only)

    Probabilistic traffic splitting to prevent collapse

    Stability guards to avoid oscillations

Why This Matters

This transforms the project from:

    “traffic prediction”
    into
    “intelligent transportation control system”



