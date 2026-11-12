# preprocessing (Feature Scalers)

This part of the documentation contains complete information about data preprocessing and scaling modules in `rslearn`.

* <a href="../../README.md">Back to docs/</a>
* <a href="../README.md">Back to user_guide</a>

---

## Overview

Feature scaling is a vital preprocessing step for machine learning algorithms, particularly gradient-based optimization algorithms (such as `LinearRegression`, `LogisticRegression`, `Ridge`, `Lasso`, `ElasticNet`) and distance-based algorithms (such as `KNNClassifier`, `KNNRegressor`).

`rslearn` provides two primary scaling classes implemented in pure NumPy:
1. **`StandardScaler`**: Centers features by subtracting the mean and scaling to unit variance.
2. **`MinMaxScaler`**: Rescales features to a bounded range, typically $[0, 1]$.

---

## Scaler Comparison

| Scaler | Mathematical Formula | Default Output Range | Outlier Sensitivity | Recommended Use Cases |
|---|---|---|---|---|
| [`StandardScaler`](standard_scaler.md) | $z = \frac{x - \mu}{\sigma}$ | Unbounded (mean 0, variance 1) | Moderately robust | Gradient descent optimization, linear models, general datasets |
| [`MinMaxScaler`](minmax_scaler.md) | $x_{\text{scaled}} = \frac{x - x_{\min}}{x_{\max} - x_{\min}} \cdot (b - a) + a$ | Bound to $[a, b]$, default $[0, 1]$ | Highly sensitive to extreme outliers | Bounded ranges, image pixel data, neural network inputs |

---

## Common API Workflow

Both scalers adhere to a consistent, intuitive API pattern:

| Method | Parameters | Return Type | Description |
|---|---|---|---|
| `fit(data)` | `data`: array-like (`n_samples,` or `n_samples, n_features`) | `None` | Computes and stores distribution parameters (mean/std or min/max). |
| `transform(data)` | `data`: array-like (`n_samples,` or `n_samples, n_features`) | `np.ndarray` | Transforms input data using previously computed parameters. |
| `fit_transform(data)` | `data`: array-like (`n_samples,` or `n_samples, n_features`) | `np.ndarray` | Fits to data and returns transformed data in a single step. |
| `inverse_transform(data)` | `data`: array-like (`n_samples,` or `n_samples, n_features`) | `np.ndarray` | Reconstructs original unscaled data from transformed values. |

---

## Table Of Content
* <a href="standard_scaler.md">StandardScaler</a> - Standardize features by removing mean and scaling to unit variance
* <a href="minmax_scaler.md">MinMaxScaler</a> - Transform features by scaling each feature to a given range
