| Metric | Penalizes Big Errors | Unit | Outlier Sensitive |
| ------ | -------------------- | ---- | ----------------- |
| MAE    | ❌ Low                | Same | ❌ Low             |
| RMSE   | ✅ High               | Same | ✅ High            |
| MAPE   | ❌ Medium             | %    | ⚠️ Zero issue     |

🚦 For Traffic / ST-Mamba Models

    MAE → stable & interpretable

    RMSE → highlights peak congestion errors

    MAPE → risky if flow/speed can be 0

👉 Most traffic papers report MAE + RMSE together


1️⃣ MAE — Mean Absolute Error
Formula
MAE=1N∑i=1N∣yi−y^i∣
MAE=N1​i=1∑N​∣yi​−y^​i​∣
Meaning

    Average of absolute errors

    Treats all errors equally

    Same unit as the target (e.g., vehicles, speed)

Pros

✔ Easy to understand
✔ Robust to outliers (compared to RMSE)
Cons

✖ Doesn’t penalize large errors strongly
Example

Actual: [100, 120, 130]
Predicted: [90, 125, 140]

Errors: |10|, |5|, |10|
MAE = (10 + 5 + 10)/3 = 8.33
2️⃣ RMSE — Root Mean Squared Error
Formula
RMSE=1N∑i=1N(yi−y^i)2
RMSE=N1​i=1∑N​(yi​−y^​i​)2
​
Meaning

    Squares errors → large errors matter more

    Sensitive to outliers

Pros

✔ Strongly penalizes big mistakes
✔ Smooth gradients (good for training)
Cons

✖ Can be dominated by a few large errors
Example

Errors: 10, -5, -10
Squares: 100, 25, 100
RMSE = √(225/3) = √75 ≈ 8.66
3️⃣ MAPE — Mean Absolute Percentage Error
Formula
MAPE=100N∑i=1N∣yi−y^iyi∣
MAPE=N100​i=1∑N​
​yi​yi​−y^​i​​
​
Meaning

    Error as a percentage

    Scale-independent

Pros

✔ Easy to interpret
✔ Good for comparison across datasets
Cons ⚠️

✖ Fails when actual value = 0
✖ Biased toward low actual values
Example

Actual: 100, Predicted: 90
MAPE = |10/100| × 100 = 10%
