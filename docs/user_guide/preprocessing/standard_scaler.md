# preprocessing.StandardScaler

## Overview
`StandardScaler` standardizes features by removing the mean and scaling to unit variance. Each feature is centered around zero with a standard deviation of one. This is the primary scaler used across `rslearn`'s internal auto-scaling routines.

**Features**  
* Automatically handles both 1D and 2D arrays, lists, tuples, and pandas DataFrames.
* Zero-division protection: features with zero variance (constant columns) are assigned a standard deviation of 1.
* Fully reversible transformations via `inverse_transform()`.
* Pure NumPy implementation for optimal execution speed.

---

## Parameters

### Class Parameter (tuning parameters)

`StandardScaler` does not take required configuration parameters during initialization.

| Parameter | Type | Default | Description |
|---|---|---|---|
| None | - | - | Standard initialization requires no extra arguments. |

**Usage**  
```python
from rslearn.preprocessing import StandardScaler

scaler = StandardScaler()
```

---

## NOTE
1. **Preventing Data Leakage**: Always fit `StandardScaler` **only on training data** (`X_train`), and use `.transform()` on testing data (`X_test`). Never fit on the entire dataset prior to splitting.
2. **Auto-scaling in Models**: Models such as `LinearRegression`, `LogisticRegression`, `Ridge`, `Lasso`, `ElasticNet`, and KNN enable this scaler automatically when `scale=True` (and `hard_scale_off=False`).

---

## Mathematical Formulation

For each feature $j$:
$$\mu_j = \frac{1}{N} \sum_{i=1}^N X_{i,j}$$

$$\sigma_j = \sqrt{\frac{1}{N} \sum_{i=1}^N (X_{i,j} - \mu_j)^2}$$

The standardized value is:
$$z_{i,j} = \frac{X_{i,j} - \mu_j}{\sigma_j}$$

Reconstruction via `inverse_transform`:
$$X_{i,j} = z_{i,j} \cdot \sigma_j + \mu_j$$

---

### Method Parameters

| Method | Parameters | Return Type | Description |
|---|---|---|---|
| `fit(data)` | `data`: array-like (`n_samples, n_features` or `n_samples,`) | `None` | Computes the mean ($\mu$) and standard deviation ($\sigma$) across samples for each feature column. |
| `transform(data)` | `data`: array-like (`n_samples, n_features` or `n_samples,`) | `np.ndarray` | Scales `data` using fitted mean and standard deviation. Raises `NotFittedError` if not fitted. |
| `fit_transform(data)` | `data`: array-like (`n_samples, n_features` or `n_samples,`) | `np.ndarray` | Combines `fit()` and `transform()` into a single efficient operation. |
| `inverse_transform(data)` | `data`: array-like (`n_samples, n_features` or `n_samples,`) | `np.ndarray` | Reverses the scaling operation back to the original feature space. |

---

## Usage

### Complete Practical Example

```python
import numpy as np
from rslearn.preprocessing import StandardScaler
from rslearn.model_selection import train_test_split

# 1. Create sample dataset with features on different scales
X = np.array([
    [10.0, 10000.0],
    [20.0, 25000.0],
    [15.0, 18000.0],
    [30.0, 50000.0],
    [25.0, 32000.0],
    [18.0, 22000.0]
])
y = np.array([100, 240, 180, 490, 310, 210])

# 2. Split dataset first to avoid data leakage
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# 3. Initialize StandardScaler
scaler = StandardScaler()

# 4. Fit and transform training data
X_train_scaled = scaler.fit_transform(X_train)
print("Scaled Training Features:")
print(X_train_scaled)
print(f"Computed Means: {scaler.mean}")
print(f"Computed Std:   {scaler.std}")

# 5. Transform testing data using the fitted scaler
X_test_scaled = scaler.transform(X_test)
print("Scaled Test Features:")
print(X_test_scaled)

# 6. Reverse transform back to original values
X_restored = scaler.inverse_transform(X_train_scaled)
print("Restored Features (matches original X_train):")
print(np.round(X_restored, 2))
```

---

# Follow up
* [MinMaxScaler](minmax_scaler.md) - Bound features to a specific range (e.g., $[0, 1]$)
* [LinearRegression](../linear_model/linear_regression.md) - Uses StandardScaler by default
* [train_test_split](../model_selection/splitting.md) - Prepare train and test splits
