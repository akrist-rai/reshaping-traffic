## 🔄 Data Flow

```text
Raw Data (np.ndarray)
        ↓
Dataset
  ├─ Sliding Window (history_len, horizon)
  └─ Normalization (mean, std)
        ↓
DataLoader
  ├─ Batching
  └─ Shuffling
        ↓
Model Input Reshape
        ↓
Model Forward Pass
        ↓
Prediction
