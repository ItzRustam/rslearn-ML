# preprocessing.MinMaxScaler

## Overview
`MinMaxScaler` transforms features by scaling each feature individually to a bounded range, typically between 0 and 1. It preserves the exact shape of the original distribution and keeps zero entries intact in sparse distributions.

**Features**  
* Configurable output range via `feature_range=(min, max)`.
* Compatible with 1D and 2D arrays, lists, tuples, and pandas DataFrames.
* Automatic zero-division prevention for constant feature columns.
* Invertible transformation via `inverse_transform()`.

---

## Parameters

### Class Parameter (tuning parameters)

| Parameter | Type | Default | Description |
|---|---|---|---|
| `feature_range` | `tuple (float, float)` | `(0, 1)` | Desired range of transformed data $(a, b)$ where $a < b$. |

**Usage**  
```python
from rslearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
```

---

## NOTE
1. **Outlier Sensitivity**: `MinMaxScaler` is sensitive to outliers because extreme values will compress the majority of inlier data points into a very narrow sub-range. For datasets with significant outliers, consider `StandardScaler`.
2. **Preventing Data Leakage**: Call `fit()` or `fit_transform()` exclusively on `X_train`, then call `transform()` on `X_test`.

---

## Mathematical Formulation

Given a target range $[a, b]$, where $a$ is the minimum desired value and $b$ is the maximum:

$$X_{\text{std}} = \frac{X - X_{\min}}{X_{\max} - X_{\min}}$$

$$X_{\text{scaled}} = X_{\text{std}} \cdot (b - a) + a$$

To reconstruct original unscaled data via `inverse_transform`:

$$X_{\text{orig}} = \frac{X_{\text{scaled}} - a}{b - a} \cdot (X_{\max} - X_{\min}) + X_{\min}$$

---

### Method Parameters

| Method | Parameters | Return Type | Description |
|---|---|---|---|
| `fit(X)` | `X`: array-like (`n_samples, n_features` or `n_samples,`) | `None` | Computes the minimum ($X_{\min}$) and maximum ($X_{\max}$) values per feature column. |
| `transform(X)` | `X`: array-like (`n_samples, n_features` or `n_samples,`) | `np.ndarray` | Rescales data to the target `feature_range`. Raises `NotFittedError` if uninitialized. |
| `fit_transform(X)` | `X`: array-like (`n_samples, n_features` or `n_samples,`) | `np.ndarray` | Fits parameters and transforms input data in a single step. |
| `inverse_transform(X)` | `X`: array-like (`n_samples, n_features` or `n_samples,`) | `np.ndarray` | Reverses the scaling back to the original feature coordinates. |

---

## Usage

### Complete Practical Example

```python
import numpy as np
from rslearn.preprocessing import MinMaxScaler
from rslearn.model_selection import train_test_split

# 1. Sample feature dataset with diverse column magnitudes
X = np.array([
    [120.0, 0.05],
    [340.0, 0.95],
    [210.0, 0.40],
    [500.0, 1.20],
    [150.0, 0.15],
    [450.0, 0.85]
])
y = np.array([12, 35, 22, 51, 16, 46])

# 2. Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# 3. Initialize MinMaxScaler to scale features between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))

# 4. Fit on train set and transform
X_train_scaled = scaler.fit_transform(X_train)
print("MinMax Scaled Train Data (values in [0, 1]):")
print(X_train_scaled)

# 5. Transform test set with the fitted scaler
X_test_scaled = scaler.transform(X_test)
print("MinMax Scaled Test Data:")
print(X_test_scaled)

# 6. Revert scaled data to original values
X_train_reconstructed = scaler.inverse_transform(X_train_scaled)
print("Reconstructed Train Data:")
print(np.round(X_train_reconstructed, 2))
```

---

# Follow up
* [StandardScaler](standard_scaler.md) - Standardize features with mean 0 and variance 1
* [LinearRegression](../linear_model/linear_regression.md) - Linear regression model
* [KNNClassifier](../neighbors/KNNClassifier.md) - K-Nearest Neighbors classifier
