

````md
## 🔀 DataLoader Shuffling Explained

In this project, **data shuffling happens at the DataLoader level**, not at the raw data level.

---

### ❌ What DataLoader Does NOT Do

- Does **not** shuffle the original `np.ndarray`
- Does **not** modify time steps
- Does **not** regenerate sliding windows
- Does **not** change normalization statistics

The raw data and window boundaries remain **fixed and deterministic**.

---

### ✅ What DataLoader Actually Does

The `Dataset` first converts the raw `np.ndarray` into **sliding-window samples**.

```text
dataset[0] → window #0
dataset[1] → window #1
dataset[2] → window #2
...
````

When `shuffle=True` is enabled in the `DataLoader`, it:

* Shuffles **dataset indices**
* Changes the **order in which windowed samples are served**
* Preserves temporal structure *inside* each window

```text
Original order:
[0, 1, 2, 3, 4, 5]

Shuffled order:
[4, 1, 5, 0, 3, 2]
```

The DataLoader then fetches samples as:

```python
dataset[4], dataset[1], dataset[5], dataset[0], ...
```

---

### 🧠 Key Insight

> **Sliding windows are created once by the Dataset.
> DataLoader only changes the order in which those windows are accessed.**

---

### 📦 Why Shuffling Is Important

* Prevents gradient correlation from sequential samples
* Improves training stability
* Reduces temporal overfitting
* Maintains correct temporal context within each sample

---

### ⚠️ Validation & Testing

For validation and test sets:

```python
shuffle = False
```

This preserves chronological order for correct evaluation.



### ✅ One-Line Summary

> **DataLoader shuffles dataset indices, not the underlying time-series data or sliding-window structure.**

```

