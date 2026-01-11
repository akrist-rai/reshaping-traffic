
---

````md
# STMambaBlock: Interleaved Spatio-Temporal Modeling

This module implements an **Interleaved Spatio-Temporal Block** that explicitly separates **spatial** and **temporal** reasoning using specialized architectures:

- **GAT (Graph Attention Network)** for spatial dependencies
- **Mamba (State Space Model)** for temporal dynamics

Rather than jointly modeling space and time in a single operation, this block **factorizes the problem** and alternates between dimensions using tensor pivots.

---

## High-Level Overview

**Input shape**
```text
[Batch, Time, Nodes, Features]
````

**Processing flow**

```text
Input
  ↓
Spatial Modeling (GAT across Nodes)
  ↓
Pivot (reinterpret Nodes ↔ Time)
  ↓
Temporal Modeling (Mamba across Time)
  ↓
Pivot Back
  ↓
Output
```

This design allows each module to operate on the structure it is best suited for.

---

## Core Design Principle

> The same data is not merely passed through layers —
> it is **reinterpreted across dimensions** so that different inductive biases can act on it.

* GAT assumes **graph-structured spatial relationships**
* Mamba assumes **sequential temporal dynamics**
* Tensor pivots allow both assumptions to hold without information loss

---

## Detailed Phase Breakdown

---

## Phase 1: Spatial Modeling (GAT)

```python
x_slices = x.view(B * T, N, F)
x_spatial = self.spatial_layer(x_slices, adj_matrix)
x_spatial = self.norm_spatial(x_slices + dropout(x_spatial))
```

### Interpretation

* Each **time step is processed independently**
* Nodes attend to neighboring nodes using the adjacency matrix
* Time is treated as a batch dimension

Mathematically:
[
\mathbf{h}*i^t = \sum_j \alpha*{ij} \mathbf{h}_j^t
]

### Key Properties

* Captures **spatial correlations**
* Prevents temporal leakage
* Residual connection avoids over-smoothing

---

## Phase 2: Pivot (Dimension Rotation)

```python
x_rotated = x.permute(0, 2, 1, 3)
x_temporal_input = x_rotated.reshape(B * N, T, F)
```

### What This Does

The pivot changes the **meaning** of the sequence dimension:

| Before Pivot       | After Pivot         |
| ------------------ | ------------------- |
| Sequence = Nodes   | Sequence = Time     |
| Independent = Time | Independent = Nodes |

Each node now has its **own temporal sequence**.

📌 This operation **does not modify values** — only their interpretation.

---

## Phase 3: Temporal Modeling (Mamba)

```python
x_temporal_out = self.temporal_layer(x_temporal_input)
x_temporal = self.norm_temporal(
    x_temporal_input + dropout(x_temporal_out)
)
```

### Interpretation

* Mamba models **long-range temporal dependencies**
* Each node evolves independently through time
* Internal state captures historical context

Mathematically:
[
\mathbf{h}_i^{t+1} = f(\mathbf{h}_i^t, \mathbf{s}_i)
]

### Why Mamba?

* Linear-time complexity
* Stable for long sequences
* Better inductive bias than attention for time series

---

## Phase 4: Restore Original Geometry

```python
x_restored = x_temporal.view(B, N, T, F)
x_final = x_restored.permute(0, 2, 1, 3).contiguous()
```

The tensor is returned to its original layout:

```text
[Batch, Time, Nodes, Features]
```

Now enriched with:

* Spatial context (from GAT)
* Temporal memory (from Mamba)

---

## Why This Is NOT Just Layer Stacking

### ❌ What This Is Not

* Simple data forwarding
* Plain sequential layers
* Joint space-time attention

### ✅ What This Is

* Explicit **space–time factorization**
* Alternating inductive biases
* Two distinct computational graphs on the same tensor
* Residual spatio-temporal fusion

---

## Inductive Bias Separation

| Component | Learns                      | Why It Fits           |
| --------- | --------------------------- | --------------------- |
| GAT       | Spatial relationships       | Graph topology        |
| Mamba     | Temporal evolution          | Sequential dynamics   |
| Pivot     | Structural reinterpretation | No information loss   |
| Residuals | Stability                   | Gradient preservation |

---

## Advantages of This Design

* Avoids quadratic attention over space–time
* Scales to long temporal horizons
* Prevents spatial over-smoothing
* Improves training stability
* Modular and extensible

---

## Summary Insight

> **This block does not simply pass data forward —
> it repeatedly reprojects the same signal into different structural views so that spatial and temporal reasoning can be learned independently and efficiently.**

---

## Output Shape

```text
Input : [B, T, N, F]
Output: [B, T, N, F]
```

Shape-preserving, context-enriched transformation.

---

## Suitable Use Cases

* Traffic forecasting
* Spatio-temporal graphs
* Sensor networks
* Mobility modeling
* Long-horizon time series on graphs

---

```

---

If you want next:
- Add **mathematical proofs**
- Add **complexity analysis**
- Add **ablation study notes**
- Add **diagram figures (for papers)**

Just tell me 👍
```
