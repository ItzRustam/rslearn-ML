# neighbors.KNNRegressor

## Overview
`KNNRegressor` is a non-parametric supervised regression algorithm. Given an unseen feature vector, it computes Euclidean distances to all training points, identifies the $k$ nearest neighbors, and predicts the target value as the arithmetic mean of their target values.

**Features**  
* Supports both 1D and multi-feature continuous regression.
* Automatic feature scaling with `StandardScaler` during `fit(scale=True)`.
* Full user control over scaling using `hard_scale_off=True`.
* Built-in model evaluation returning $R^2$, MSE, MAE, and RMSE.
* Compatible with Python lists, tuples, pandas DataFrames, and NumPy arrays.

---

## Parameters

### Class Parameter (tuning parameters)

| Parameter | Type | Default | Description |
|---|---|---|---|
| `k_neighbors` | `int` | `5` | Number of nearest neighbors to average for regression predictions. |
| `hard_scale_off` | `bool` | `False` | When `True`, internal auto-scaling is bypassed and data is used unscaled. |

**Usage**  
```python
from rslearn.neighbors import KNNRegressor

model = KNNRegressor(k_neighbors=5, hard_scale_off=False)
```

---

## NOTE
1. **Feature Scaling**: When features have different physical units or ranges, Euclidean distance is disproportionately influenced by the feature with the largest magnitude. Keep `hard_scale_off=False` and `scale=True` unless you have scaled your dataset beforehand.
2. **Choice of $k$**: Smaller values of $k$ (e.g., $k=1, 2$) make the model more sensitive to noise and outliers (higher variance). Larger values of $k$ smooth the prediction function (higher bias).

---

### Method Parameters

| Method | Parameters | Return Type | Description |
|---|---|---|---|
| `fit(X, y, scale=True)` | `X`: Feature matrix (`n_samples, n_features`)<br>`y`: Target vector (`n_samples,`)<br>`scale`: `bool`, default `True` | `None` | Stores training samples and computes scaling statistics when `scale=True` and `hard_scale_off=False`. |
| `predict(X_new)` | `X_new`: Query feature matrix (`n_samples, n_features`) | `np.ndarray` | Calculates distances, finds $k$ nearest neighbors for each sample, and returns predicted target means as `np.ndarray`. |
| `evaluate(X=None, y_pred=None, y_true=None)` | `X`: Optional feature matrix for prediction<br>`y_pred`: Optional precomputed predictions<br>`y_true`: Ground truth target values (`n_samples,`) | `dict` | Calculates regression metrics (`r2_score`, `mse`, `mae`, `rmse`) and returns a dictionary with `{"prediction": ..., "evaluation": ...}`. |

---

## Usage

### Complete End-to-End Example

```python
import numpy as np
from rslearn.neighbors import KNNRegressor
from rslearn.model_selection import train_test_split

# 1. Generate synthetic continuous regression dataset
np.random.seed(42)
X = np.random.uniform(1.0, 10.0, size=(100, 3))
# Target is a continuous function of features plus some noise
y = 2.5 * X[:, 0] - 1.8 * X[:, 1] + 3.2 * X[:, 2] + np.random.normal(0, 0.5, size=100)

# 2. Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Initialize and fit KNNRegressor
model = KNNRegressor(k_neighbors=5, hard_scale_off=False)
model.fit(X_train, y_train, scale=True)

# 4. Predict target values for test set
predictions = model.predict(X_test)
print(f"Sample Predictions: {predictions[:5]}")

# 5. Evaluate model performance
evaluations = model.evaluate(y_pred=predictions, y_true=y_test)
print("Regression Evaluation Metrics:")
print(f"R² Score: {evaluations['evaluation']['r2_score']:.4f}")
print(f"MSE:      {evaluations['evaluation']['mse']:.4f}")
print(f"MAE:      {evaluations['evaluation']['mae']:.4f}")
print(f"RMSE:     {evaluations['evaluation']['rmse']:.4f}")
```

---

# Follow up
* [KNNClassifier](KNNClassifier.md) - For categorical classification tasks
* [LinearRegression](../linear_model/linear_regression.md) - Parametric linear regression model
* [StandardScaler](../preprocessing/standard_scaler.md) - Feature standardization
* [Regression Metrics](../metrics/regression.md) - R², MSE, MAE, and RMSE details
* [evaluate_model](../metrics/evaluate_model.md) - Multi-metric evaluation utility
