# metrics.regression

## Overview
The `regression` module in `rslearn.metrics` implements standard regression loss and goodness-of-fit metrics from scratch using NumPy. It supports both single-target (1D) and multi-output (2D) regression tasks, with optional custom weighting and output aggregation.

**Features**  
* Supports single-output `(n_samples,)` and multi-output `(n_samples, n_outputs)`.
* Flexible output aggregation: `uniform_average`, `weighted`, and `raw_values`.
* Built-in numerical stability safeguards (e.g. constant target $R^2$ handling).
* Compatible with Python lists, tuples, and NumPy arrays.

---

## Common Parameters

All regression metrics share the following core parameters:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `y_true` | `array-like` of shape `(n_samples,)` or `(n_samples, n_outputs)` | *Required* | Ground truth (correct) target values. |
| `y_pred` | `array-like` of shape `(n_samples,)` or `(n_samples, n_outputs)` | *Required* | Estimated target values returned by a regressor. |
| `multi_output` | `str` | `"uniform_average"` | How to aggregate scores across multi-output dimensions (`"uniform_average"`, `"weighted"`, `"raw_values"`). |
| `weights` | `array-like` of shape `(n_outputs,)` | `None` | Output weights used when `multi_output="weighted"`. |

---

## Available Regression Metrics

| Metric Function | Formula | Best Value | Description |
|---|---|---|---|
| [`r2_score`](#1-r2_score) | $1 - \frac{\sum (y - \hat{y})^2}{\sum (y - \bar{y})^2}$ | `1.0` | Coefficient of Determination; proportion of variance explained. |
| [`mse`](#2-mse) | $\frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2$ | `0.0` | Mean Squared Error; average squared difference. |
| [`mae`](#3-mae) | $\frac{1}{N} \sum_{i=1}^N \|y_i - \hat{y}_i\|$ | `0.0` | Mean Absolute Error; average absolute difference. |
| [`rmse`](#4-rmse) | $\sqrt{\frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2}$ | `0.0` | Root Mean Squared Error; square root of MSE. |

---

## Function Details & Usage

### 1. `r2_score`

Computes the coefficient of determination ($R^2$). A score of $1.0$ indicates perfect predictions, $0.0$ indicates performance equal to a constant model predicting the mean, and negative values indicate worse-than-mean predictions.

#### Mathematical Formula
$$R^2 = 1 - \frac{\sum_{i=1}^N (y_i - \hat{y}_i)^2}{\sum_{i=1}^N (y_i - \bar{y})^2}$$

#### Usage
```python
from rslearn.metrics import r2_score

y_true = [3.0, -0.5, 2.0, 7.0]
y_pred = [2.5, 0.0, 2.0, 8.0]

score = r2_score(y_true, y_pred)
print(f"R² Score: {score:.4f}")
```

---

### 2. `mse` (Mean Squared Error)

Calculates the average of the squared prediction errors. Highly sensitive to outliers due to the squaring of residual errors.

#### Mathematical Formula
$$\text{MSE} = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2$$

#### Usage
```python
from rslearn.metrics import mse

y_true = [3.0, -0.5, 2.0, 7.0]
y_pred = [2.5, 0.0, 2.0, 8.0]

loss = mse(y_true, y_pred)
print(f"MSE: {loss:.4f}")
```

---

### 3. `mae` (Mean Absolute Error)

Calculates the average magnitude of the errors without considering their direction. It provides a linear penalty and is more robust to extreme outliers than MSE.

#### Mathematical Formula
$$\text{MAE} = \frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|$$

#### Usage
```python
from rslearn.metrics import mae

y_true = [3.0, -0.5, 2.0, 7.0]
y_pred = [2.5, 0.0, 2.0, 8.0]

error = mae(y_true, y_pred)
print(f"MAE: {error:.4f}")
```

---

### 4. `rmse` (Root Mean Squared Error)

Computes the square root of the Mean Squared Error. Expresses error in the same units as the target variable.

#### Mathematical Formula
$$\text{RMSE} = \sqrt{\text{MSE}} = \sqrt{\frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2}$$

#### Usage
```python
from rslearn.metrics import rmse

y_true = [3.0, -0.5, 2.0, 7.0]
y_pred = [2.5, 0.0, 2.0, 8.0]

root_error = rmse(y_true, y_pred)
print(f"RMSE: {root_error:.4f}")
```

---

## Multi-Output Regression Examples

```python
import numpy as np
from rslearn.metrics import r2_score, mse, mae, rmse

# Multi-output targets (2 target columns per sample)
y_true = np.array([
    [3.0, 10.0],
    [-0.5, 20.0],
    [2.0, 30.0],
    [7.0, 40.0]
])

y_pred = np.array([
    [2.5, 12.0],
    [0.0, 18.0],
    [2.0, 33.0],
    [8.0, 39.0]
])

# 1. Uniform Average across both outputs (default)
print("Uniform Average R²:", r2_score(y_true, y_pred))
print("Uniform Average MSE:", mse(y_true, y_pred))

# 2. Raw values for each output column
print("Raw R² per column:", r2_score(y_true, y_pred, multi_output="raw_values"))

# 3. Weighted Average (giving 70% weight to output 1, 30% to output 2)
print("Weighted MSE:", mse(y_true, y_pred, multi_output="weighted", weights=[0.7, 0.3]))
```

---

# Follow up
* [Classification Metrics](classification.md) - Accuracy, Precision, Recall, and F1
* [evaluate_model](evaluate_model.md) - Single function returning all metrics
* [LinearRegression](../linear_model/linear_regression.md) - Linear regression estimator
* [KNNRegressor](../neighbors/KNNRegressor.md) - K-Nearest Neighbors regressor
