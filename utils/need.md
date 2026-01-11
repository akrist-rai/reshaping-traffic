```python
def masked_mae(pred, true, eps=1e-5):
    mask = (true != 0).float()
    loss = torch.abs(pred - true)
    return (loss * mask).sum() / (mask.sum() + eps)
```

## 🔴 The core problem this function solves

> **Some values in `true` are NOT real data.**
> They are just **placeholders (0)**.

If you calculate normal MAE, your model gets **punished for predicting values where no real data exists**.

That is **WRONG training**.

---

## ❌ What goes wrong WITHOUT `masked_mae`

### Example (realistic)

You are doing **traffic prediction** 🚗
Some sensors are **offline** → their value is stored as `0`.

```python
true = [60, 55, 0, 70, 0]   # 0 = sensor OFF
pred = [58, 54, 40, 69, 30]
```

### Normal MAE (WRONG)

```python
abs(pred - true)
= [2, 1, 40, 1, 30]

MAE = (2 + 1 + 40 + 1 + 30) / 5 = 14.8 ❌
```

⚠️ Your model is punished for:

* predicting **40** where sensor was OFF
* predicting **30** where sensor was OFF

But **there was no real ground truth** there.

---

## ✅ What `masked_mae` does (CORRECT)

```python
mask = (true != 0)
mask = [1, 1, 0, 1, 0]
```

Apply mask:

```python
errors = [2, 1, 40, 1, 30]
masked = [2, 1, 0, 1, 0]
```

Masked MAE:

```python
(2 + 1 + 1) / 3 = 1.33 ✅
```

🔥 Now the model is judged **ONLY on real data**.

---

## 🎯 So what purpose does this serve?

### 1️⃣ Ignores fake / missing values

Zeros are **not measurements**, just placeholders.

---

### 2️⃣ Prevents wrong gradients

Without masking:

* Model learns: “predict 0 everywhere”
* Training becomes unstable
* Performance looks worse than it is

---

### 3️⃣ Makes evaluation FAIR

You evaluate only where:

* sensor exists
* data is meaningful

---

### 4️⃣ Absolutely required in:

* Traffic forecasting (METR-LA, PeMS)
* Time-series with missing steps
* Sensor networks
* Graph neural networks
* Padded sequences

---

## 🧠 One-line intuition (remember this)

> **Masked MAE = MAE, but only where ground truth actually exists**

---

## 🧪 Why not just delete zeros?

Because:

* Tensors must stay same shape (GPU batch)
* Graph/time alignment would break
* Masking is faster than slicing

---

## 🧑‍💻 CS / DSA analogy

Like this logic:

```cpp
sum = 0; count = 0;
for (i = 0; i < n; i++) {
    if (true[i] != 0) {
        sum += abs(pred[i] - true[i]);
        count++;
    }
}
ans = sum / count;
```

🔥 **Masked MAE does this in ONE GPU operation**.

---

## 🚨 Key takeaway

If you remove `mask`:
❌ model learns wrong
❌ evaluation lies
❌ loss explodes
❌ papers get rejected

If you keep it:
✅ fair training
✅ correct gradients
✅ standard practice in ML research

