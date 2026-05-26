# 📊 rslearn.metrics

This module provides essential regression evaluation metrics implemented from scratch using NumPy.

Currently, it includes:

- **R² Score (`r2_score`)**
- **mean_squared_error (`mse`)**
- **Mean Absolute Error (`mse`)** 
- **Root Mean Squared Error (`rmse`)**

### Visit For classification Evaluation Metics
**[classification tools](classification_readme.md)**


---

## 🚀 Features

- Supports both **1D and multi-output regression**
- Handles:
  - `(n_samples,)`
  - `(n_samples, n_outputs)`
- Built with clean API design inspired by `scikit-learn`
- Supports:
  - Uniform averaging
  - Weighted averaging
  - Raw output scores
  - list and numpy array both supported

---

## 📌 Available Functions

# 🔹 `r2_score`

Compute the **coefficient of determination (R² score)**.

**Formula**
**`sum of (y_true-y_pred)^2`** / **`sum of (y_true - mean(y_true))^2`**


| Parameter      | Description                     |
| -------------- | ------------------------------- |
| `y_true`       | Ground truth values             |
| `y_pred`       | Predicted values                |
| `multi_output` | How to handle multi-output data |
| `weights`      | Weights for weighted averaging  |

#### `multi_output` options
| Option            | Description                             |
| ----------------- | --------------------------------------- |
| `uniform_average` | Average scores across outputs (default) |
| `weighted`        | Weighted average using `weights`        |
| `raw_values`      | Return score for each output            |

#### Return Type
**`float` between 0.0 to 1.0**

#### How to Use

- `For 1D array`
```python
from rslearn.metrics import r2_score

y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]

score = r2_score(y_true, y_pred)
print(score)
```

- `For Multi Ouput Array`
```python
y_true = [[3, 10], [-0.5, 20], [2, 30], [7, 40]]
y_pred = [[2.5, 12], [0.0, 18], [2, 33], [8, 39]]

score = r2_score(y_true, y_pred)
print(score)
```

- `For Multi Ouput Array with Weights`
``` python
score = r2_score(
    y_true,
    y_pred,
    multi_output="weighted",
    weights=[0.6, 0.4]
)
```

#### ⚠️ Notes
- Input shapes must match  
- Minimum 2 samples required  
- For constant y_true, R² returns 0.0

----------

# 🔹 `MSE`
Compute the Mean Squred Error  
**Formula**
**`sum of (y_true - y_pred)^2`** / **`n`**
- `n = length of y_true`

| Parameter      | Description                     |
| -------------- | ------------------------------- |
| `y_true`       | Ground truth values             |
| `y_pred`       | Predicted values                |
| `multi_output` | How to handle multi-output data |
| `weights`      | Weights for weighted averaging  |

#### `multi_output` options
| Option            | Description                             |
| ----------------- | --------------------------------------- |
| `uniform_average` | Average errors across outputs (default) |
| `weighted`        | Weighted average using `weights`        |
| `raw_values`      | Return error for each output            |

#### Return Type
**`float` any to any**

#### How to Use

- `For 1D array`
```python
from rslearn.metrics import mse

y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]

error = mse(y_true, y_pred)
print(error)
```

- `For Multi Ouput Array`
```python
y_true = [[3, 10], [-0.5, 20], [2, 30], [7, 40]]
y_pred = [[2.5, 12], [0.0, 18], [2, 33], [8, 39]]

error = mse(y_true, y_pred)
print(error)
```

- `For Multi Ouput Array with Weights`
``` python
error = mse(
    y_true,
    y_pred,
    multi_output="weighted",
    weights=[0.6, 0.4]
)
```

#### ⚠️ Notes
- Input shapes must match

---------------

# 🔹 `MAE`
Compute the Mean Absolute Error | Better for Outliers than MSE  
**Formula**
**`sum of | y_true - y_pred |`** / **`n`**
- `n = length of y_true`

| Parameter      | Description                     |
| -------------- | ------------------------------- |
| `y_true`       | Ground truth values             |
| `y_pred`       | Predicted values                |
| `multi_output` | How to handle multi-output data |
| `weights`      | Weights for weighted averaging  |

#### `multi_output` options
| Option            | Description                             |
| ----------------- | --------------------------------------- |
| `uniform_average` | Average errors across outputs (default) |
| `weighted`        | Weighted average using `weights`        |
| `raw_values`      | Return error for each output            |

#### Return Type
**`float` any to any**

#### How to Use

- `For 1D array`
```python
from rslearn.metrics import mae

y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]

error = mae(y_true, y_pred)
print(error)
```

- `For Multi Ouput Array`
```python
y_true = [[3, 10], [-0.5, 20], [2, 30], [7, 40]]
y_pred = [[2.5, 12], [0.0, 18], [2, 33], [8, 39]]

error = mae(y_true, y_pred)
print(error)
```

- `For Multi Ouput Array with Weights`
``` python
error = mae(
    y_true,
    y_pred,
    multi_output="weighted",
    weights=[0.6, 0.4]
)
```

#### ⚠️ Notes
- Input shapes must match

-----------

# 🔹 `RMSE`
Compute the Root Mean Squred Error   
**Formula**
**`Sqaure root of 'MSE'`**

| Parameter      | Description                     |
| -------------- | ------------------------------- |
| `y_true`       | Ground truth values             |
| `y_pred`       | Predicted values                |
| `multi_output` | How to handle multi-output data |
| `weights`      | Weights for weighted averaging  |

#### `multi_output` options
| Option            | Description                             |
| ----------------- | --------------------------------------- |
| `uniform_average` | Average errors across outputs (default) |
| `weighted`        | Weighted average using `weights`        |
| `raw_values`      | Return error for each output            |

#### Return Type
**`float` any to any**

#### How to Use

- `For 1D array`
```python
from rslearn.metrics import rmse

y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]

error = rmse(y_true, y_pred)
print(error)
```

- `For Multi Ouput Array`
```python
y_true = [[3, 10], [-0.5, 20], [2, 30], [7, 40]]
y_pred = [[2.5, 12], [0.0, 18], [2, 33], [8, 39]]

error = rmse(y_true, y_pred)
print(error)
```

- `For Multi Ouput Array with Weights`
``` python
error = rmse(
    y_true,
    y_pred,
    multi_output="weighted",
    weights=[0.6, 0.4]
)
```

#### ⚠️ Notes
- Input shapes must match

-----------------