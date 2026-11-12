# linear_model.LinearRegression

## Overview
`LinearRegression` is a parametric regression algorithm optimized using Gradient Descent. It models the relationship between dependent continuous targets and independent features by learning optimal weights and bias.

**Features**  
* Integrated automatic feature scaling via `StandardScaler` (`scale=True`) or `backupScaler`.
* Full user control over scaling using `hard_scale_off=True`.
* Built-in L1 (Lasso), L2 (Ridge), and ElasticNet regularization directly via the `regulization` parameter.
* Native all-in-one performance evaluation via `.evaluate()`.
* Model serialization and persistence in `.rslr` format.

---

## Parameters

### Class Parameters (tuning parameters)

| Parameter | Type | Default | Description |
|---|---|---|---|
| `regulization` | `str` or `None` | `None` | Regularization penalty type: `"l1"` (Lasso), `"l2"` (Ridge), `"elastic_net"` (ElasticNet), or `None`. |
| `alpha` | `float` | `0.1` | Regularization penalty strength for L1, L2, or ElasticNet penalties. |
| `l1_ratio` | `float` | `0.5` | Mixing parameter for ElasticNet ($0 \le \text{l1\_ratio} \le 1$). |
| `lr` | `float` | `0.001` | Learning rate for gradient descent weight updates. |
| `max_itr` | `int` | `3000` | Maximum number of gradient descent iterations during fitting. |
| `weights` | `np.ndarray` | `None` | Initial weight vector. Initialized randomly if `None`. |
| `bias` | `float` | `None` | Initial bias scalar. Initialized to 0.0 if `None`. |
| `min_loss` | `float` | `0.1` | Early stopping loss threshold for gradient descent. |
| `hard_scale_off` | `bool` | `False` | When `True`, bypasses all internal auto-scaling routines. |

**Usage**  
```python
from rslearn.linear_model import LinearRegression

model = LinearRegression(
    regulization="l1",
    alpha=0.2,
    lr=0.01,
    max_itr=5000,
    hard_scale_off=False
)
```

---

## NOTE
1. **Feature Scaling**: Because `LinearRegression` updates weights iteratively via Gradient Descent, feature scaling is strongly recommended. Leave `hard_scale_off=False` and `scale=True` unless you have already scaled your dataset using `StandardScaler` or `MinMaxScaler`.
2. **Early Stopping**: If the computed MSE loss falls below `min_loss`, training terminates early to prevent overfitting.

---

### Method Parameters

| Method | Parameters | Return Type | Description |
|---|---|---|---|
| `fit(X, y, scale=True)` | `X`: Feature matrix (`n_samples, n_features`)<br>`y`: Target vector (`n_samples,`)<br>`scale`: `bool`, default `True` | `None` | Optimizes model weights and bias using gradient descent. |
| `predict(X)` | `X`: Feature matrix (`n_samples, n_features`) | `np.ndarray` | Generates continuous predictions: $\hat{y} = X \cdot w + b$. |
| `evaluate(X=None, y_pred=None, y_true=None)` | `X`: Optional feature matrix<br>`y_pred`: Optional precomputed predictions<br>`y_true`: Target ground truth (`n_samples,`) | `dict` | Computes $R^2$, MSE, MAE, and RMSE. Returns `{"prediction": ..., "evaluation": ...}`. |
| `get_weight_bias()` | None | `tuple (np.ndarray, float)` | Returns learned `(weights, bias)`. |
| `save(file_name)` | `file_name`: `str`, default `"linear_model.rslr"` | `None` | Saves model parameters and scaler statistics into a compressed `.rslr` file. |

---

## Usage

### Complete Practical Example

```python
import numpy as np
from rslearn.linear_model import LinearRegression
from rslearn.model_selection import train_test_split

# 1. Generate synthetic housing feature data (area in sqft, bedrooms, age)
np.random.seed(42)
X = np.random.uniform(500, 3500, size=(200, 3))
# True relation: price = 150 * area + 20000 * bedrooms - 1200 * age + 50000
true_weights = np.array([150.0, 20000.0, -1200.0])
y = X.dot(true_weights) + 50000.0 + np.random.normal(0, 10000, size=200)

# 2. Partition dataset into train and test splits
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Initialize and fit LinearRegression
model = LinearRegression(
    regulization="l2",
    alpha=0.1,
    lr=0.01,
    max_itr=4000,
    hard_scale_off=False
)
model.fit(X_train, y_train, scale=True)

# 4. Generate predictions on test data
predictions = model.predict(X_test)
print("First 5 Predictions:", predictions[:5])

# 5. Evaluate all regression metrics simultaneously
evaluations = model.evaluate(X=X_test, y_true=y_test)
print("Evaluation results:")
print(f"R² Score: {evaluations['evaluation']['r2_score']:.4f}")
print(f"MSE:      {evaluations['evaluation']['mse']:.4f}")
print(f"MAE:      {evaluations['evaluation']['mae']:.4f}")
print(f"RMSE:     {evaluations['evaluation']['rmse']:.4f}")

# 6. Inspect learned parameters
weights, bias = model.get_weight_bias()
print("Trained Weights:", weights)
print("Trained Bias:", bias)

# 7. Save model to disk
model.save("my_linear_model.rslr")
print("Saved model to my_linear_model.rslr")
```

---

# Follow up
* [LogisticRegression](logistic_regression.md) - Classification with sigmoid and cross-entropy
* [Regularizations (Lasso, Ridge, ElasticNet)](regulizations.md) - Standalone regularization wrappers
* [StandardScaler](../preprocessing/standard_scaler.md) - Feature standardization
* [train_test_split](../model_selection/splitting.md) - Partitioning datasets
* [load_model](../loader/load_model.md) - Restore saved `.rslr` models
