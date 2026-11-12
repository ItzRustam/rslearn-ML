# BaseEstimators

This part of the documentation contains complete information about the foundational Object-Oriented Programming (OOP) hierarchy, base classes, and internal utilities in `rslearn.BaseEstimators`.

* <a href="../../README.md">Back to docs/</a>
* <a href="../README.md">Back to user_guide</a>

---

## Overview

`BaseEstimators` defines the abstract base classes and common infrastructure upon which all machine learning algorithms in `rslearn` are built. It provides consistent API design patterns across all estimators, standardizing:
1. **Automated Feature Scaling**: Internal dual-scaler management (`StandardScaler` and `backupScaler`).
2. **Unified Evaluation**: Automated multi-metric calculation through `._eval()`.
3. **Serialization Infrastructure**: Standardized `.save()` pipeline compressing models and scalers into `.rslr` and `.rslc` files.
4. **Data Standardization Utilities**: Array conversion, shape checking, and dimension validation routines in `_base`.

---

## Class Hierarchy

```
BaseEstimator (ABC)
├── LinearRegression (rslearn.linear_model)
└── LogisticRegression (rslearn.linear_model)

BaseEstimatorKNN (ABC)
├── KNNClassifier (rslearn.neighbors)
└── KNNRegressor (rslearn.neighbors)

BaseEstimatorRegulization
├── Lasso (rslearn.linear_model)
├── Ridge (rslearn.linear_model)
└── ElasticNet (rslearn.linear_model)

backupScaler
└── Secondary fallback normalizer [X / (max(X) + eps)]
```

---

## Classes Summary

| Class | Type | Key Attributes | Primary Subclasses | Purpose |
|---|---|---|---|---|
| `BaseEstimator` | Abstract Base Class (`ABC`) | `lr`, `max_itr`, `weights`, `bias`, `Scaler`, `backup_scaler`, `flag`, `type`, `_fitted` | `LinearRegression`, `LogisticRegression` | Base class for gradient descent-based parametric models. |
| `BaseEstimatorKNN` | Abstract Base Class (`ABC`) | `k`, `Scaler`, `backup_scaler`, `fitted_x`, `fitted_y`, `flag`, `type`, `_fitted` | `KNNClassifier`, `KNNRegressor` | Base class for memory-based nearest neighbor algorithms. |
| `BaseEstimatorRegulization` | Wrapper Class | `alpha`, `l1_ratio`, `min_loss`, `max_itr`, `hard_scale_off`, `regulization`, `model` | `Lasso`, `Ridge`, `ElasticNet` | High-level wrapper managing regularized linear models. |
| `backupScaler` | Helper Scaler Class | `maxx`, `epsilon=1e-9`, `_fitted` | Internal | Fallback normalizer used when `scale=False` to prevent gradient overflow. |

---

## Core Classes Breakdown

### 1. `BaseEstimator`
The abstract base for all gradient descent models (`LinearRegression`, `LogisticRegression`).

#### Key Responsibilities
* **Scaling Orchestration**: Provides `_scale_True(X, scaled=False)` and `_scale_False(X, scaled=False)`. When `scale=True`, it trains and transforms features with `StandardScaler`. When `scale=False`, it normalizes features using `backupScaler`.
* **Universal Evaluation Dispatch**: Implements `_eval(X, y_pred, y_true)` which automatically routes to `_eval_help_regression` or `_eval_help_classification` based on `self.type`.
* **Parameter Access**: Exposes `get_weight_bias()` returning `(weights, bias)`.
* **Persistence Engine**: Implements `.save()` which writes weights, biases, hyperparameters, and scaler distribution statistics to `.rslr` or `.rslc`.

---

### 2. `backupScaler`
A lightweight secondary scaling class used when user-specified feature scaling is off (`scale=False`), but internal numerical stability is still necessary to prevent gradient descent explosion.

#### Mathematical Operation
$$\text{new\_X} = \frac{X}{\max(X) + \epsilon}$$
where $\epsilon = 10^{-9}$ prevents division by zero.

#### Method Reference

| Method | Parameters | Return Type | Description |
|---|---|---|---|
| `fit(X)` | `X`: array-like | `None` | Finds and stores $\max(X)$. Marks scaler as fitted. |
| `transform(X)` | `X`: array-like | `np.ndarray` | Rescales data by dividing by $\max(X) + \epsilon$. Raises `NotFittedError` if unfitted. |
| `fit_transform(X)` | `X`: array-like | `np.ndarray` | Combines `fit()` and `transform()` in a single call. |

---

### 3. `BaseEstimatorKNN`
The abstract base class for nearest neighbor estimators (`KNNClassifier`, `KNNRegressor`).

#### Key Responsibilities
* Validates that $k \ge 1$ (`InvalidValueError` if $k \le 0$).
* Retains training data references in `self.fitted_x` and `self.fitted_y`.
* Manages feature scaling during `fit(scale=True)` and subsequent `predict()`.
* Routes `evaluate()` calls to regression or classification metrics suites.

---

### 4. `BaseEstimatorRegulization`
A delegation wrapper that standardizes parameter signatures for `Lasso`, `Ridge`, and `ElasticNet`, routing all execution to an underlying `LinearRegression` engine configured with the appropriate penalty.

---

## Internal Utility Functions (`_base.py`)

`rslearn.BaseEstimators._base` provides robust array conversion and validation tools used across `metrics`, `linear_model`, and `neighbors`:

| Utility Function | Parameters | Return Type | Purpose |
|---|---|---|---|
| `convert_array(arr1, arr2)` | `arr1`, `arr2`: array-like | `tuple (np.ndarray, np.ndarray)` | Converts inputs (tuples, lists, pandas DataFrames) to float NumPy arrays. |
| `convert1D(*args)` | `*args`: arbitrary arrays | `tuple (np.ndarray, ...)` | Flattens input arrays into 1D arrays via `np.ravel()`. |
| `shape_checker(arr1, arr2, output_mode=True)` | `arr1`, `arr2`: arrays<br>`output_mode`: `bool` | `None` | Validates that two arrays have compatible sample lengths and dimensions. |
| `check_multioutput(parameter)` | `parameter`: `str` | `None` | Validates that `multi_output` is one of `{"uniform_average", "weighted", "raw_values"}`. |
| `dim_validator(arr)` | `arr`: `np.ndarray` | `bool` | Returns `True` if array is 1D or column vector `(N, 1)`; returns `False` if multi-output `(N, D)`. |
| `multi_output_selector(multi_output_param, scores, weights=None)` | `multi_output_param`: `str`<br>`scores`: array<br>`weights`: array | `float` or `np.ndarray` | Aggregates multi-output metric scores according to the chosen strategy. |

---

## Building a Custom Estimator

Developers can create custom estimators fully compatible with `rslearn.Pipeline` and `rslearn.metrics.evaluate_model` by subclassing `BaseEstimator`:

```python
import numpy as np
from rslearn.BaseEstimators import BaseEstimator

class CustomLinearRegressor(BaseEstimator):
    def __init__(self, lr=0.01, max_itr=1000):
        super().__init__(lr=lr, max_itr=max_itr)
        self.type = "regression"      # Enables automatic regression metrics
        self._model = "CustomRegressor"

    def fit(self, X, y, scale=True):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        
        # Apply BaseEstimator auto-scaling
        if scale:
            X = self._scale_True(X, scaled=False)
            self.flag = True
        
        # Fit analytical normal equation: w = (X^T X)^-1 X^T y
        X_bias = np.c_[np.ones(X.shape[0]), X]
        beta = np.linalg.pinv(X_bias.T @ X_bias) @ X_bias.T @ y
        
        self.bias = float(beta[0])
        self.weights = beta[1:]
        self._fitted = True
        self.fitted_shape = X.shape

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if self.flag:
            X = self._scale_True(X, scaled=True)
        return X @ self.weights + self.bias

    def evaluate(self, X=None, y_pred=None, y_true=None):
        return super()._eval(X=X, y_pred=y_pred, y_true=y_true)
```

---

# Follow up
* [LinearRegression](../linear_model/linear_regression.md) - Subclass implementing gradient descent regression
* [LogisticRegression](../linear_model/logistic_regression.md) - Subclass implementing gradient descent classification
* [Pipeline](../pipeline/pipeline.md) - Pipeline executing estimators inheriting from BaseEstimator
* [Errors](../errors/README.md) - Exception classes raised during validation
