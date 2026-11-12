# linear_model.Ridge, Lasso, ElasticNet

## Overview
Regularization methods penalize large model coefficients during gradient descent optimization to mitigate overfitting, handle multicollinear features, and improve out-of-sample generalization. 

In `rslearn.linear_model`, regularized estimators are available as dedicated high-level classes—**`Lasso`**, **`Ridge`**, and **`ElasticNet`**—inheriting from `BaseEstimatorRegulization` and delegating optimization to the core `LinearRegression` engine.

**Features**  
* Prevents overfitting on training data with noisy or correlated features.
* Supports automatic feature scaling (`scale=True`) via internal `StandardScaler`.
* Full user control over feature scaling using `hard_scale_off=True`.
* Native performance evaluation through `.evaluate()`, reporting $R^2$, MSE, MAE, and RMSE.
* Automatic model persistence in compressed `.rslr` format.

---

## Regularizer Comparison

| Class | Penalty Type | Mathematical Penalty | Geometric Shape | Shrinkage Effect | Feature Selection |
|---|---|---|---|---|---|
| [`Lasso`](#1-lasso-l1-regularization) | $L_1$ Norm | $\alpha \sum_{j=1}^D \|w_j\|$ | Diamond (sharp corners) | Drives redundant coefficients exactly to zero | **Yes** (creates sparse models) |
| [`Ridge`](#2-ridge-l2-regularization) | $L_2$ Norm | $\frac{1}{2} \alpha \sum_{j=1}^D w_j^2$ | Sphere (smooth contours) | Shrinks coefficients proportionally toward zero | **No** (retains all features) |
| [`ElasticNet`](#3-elasticnet-l1--l2-regularization) | $L_1 + L_2$ Norm | $\alpha \cdot \rho \|w\|_1 + \frac{1}{2} \alpha (1 - \rho) \|w\|_2^2$ | Rounded Diamond | Combines sparsity of Lasso with stability of Ridge | **Yes** (selects groups of correlated features) |

---

## Parameters

### Class Parameters (tuning parameters)

All three regularization classes share a consistent constructor signature:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `alpha` | `float` | `0.1` | Regularization penalty strength. Larger values enforce stronger shrinkage and simpler models. |
| `l1_ratio` | `float` | `0.5` | ElasticNet mixing parameter $\rho \in [0.0, 1.0]$. At $\rho=1.0$, penalty is pure $L_1$; at $\rho=0.0$, penalty is pure $L_2$. Used primarily by `ElasticNet`. |
| `min_loss` | `float` | `0.1` | Early stopping threshold. If MSE falls below `min_loss`, gradient descent stops early. |
| `max_itr` | `int` | `3000` | Maximum number of gradient descent iterations during fitting. |
| `hard_scale_off` | `bool` | `False` | When `True`, bypasses all internal auto-scaling routines. |

**Usage**  
```python
from rslearn.linear_model import Lasso, Ridge, ElasticNet

# L1 Regularization (Lasso)
lasso_model = Lasso(alpha=0.2, max_itr=5000)

# L2 Regularization (Ridge)
ridge_model = Ridge(alpha=0.5, max_itr=5000)

# ElasticNet Regularization (L1 + L2)
enet_model = ElasticNet(alpha=0.3, l1_ratio=0.7, max_itr=5000)
```

---

## NOTE
1. **Feature Scaling is Mandatory**: Regularization penalties treat all weights uniformly. If features have unequal variance, features with larger magnitudes will be unfairly penalized. Keep `scale=True` and `hard_scale_off=False` unless the data has been pre-scaled.
2. **Underlying Architecture**: `Lasso`, `Ridge`, and `ElasticNet` wrap `LinearRegression` internally with `regulization="l1"`, `"l2"`, or `"elastic_net"`. You can also configure regularization directly through `LinearRegression(regulization="l1", alpha=0.1)`.
3. **Saving Format**: All regularized estimators save into `.rslr` files because they belong to the continuous regression family.

---

### Method Parameters

| Method | Parameters | Return Type | Description |
|---|---|---|---|
| `fit(X, y, scale=True)` | `X`: Feature matrix (`n_samples, n_features`)<br>`y`: Target array (`n_samples,`)<br>`scale`: `bool`, default `True` | `None` | Optimizes model weights and bias using regularized gradient descent. |
| `predict(X)` | `X`: Query feature matrix (`n_samples, n_features`) | `np.ndarray` | Generates continuous predictions using the learned weights and bias. |
| `evaluate(X=None, y_pred=None, y_true=None)` | `X`: Optional feature matrix<br>`y_pred`: Optional precomputed predictions<br>`y_true`: Target ground truth (`n_samples,`) | `dict` | Computes $R^2$, MSE, MAE, and RMSE. Returns `{"prediction": ..., "evaluation": ...}`. |
| `get_weight_bias()` | None | `tuple (np.ndarray, float)` | Returns learned `(weights, bias)`. |
| `save(file_name)` | `file_name`: `str`, default `"rslearn_model.rsl"` | `str` | Serializes model weights, hyperparameters, and scaler parameters to `.rslr`. |

---

## Mathematical Formulation

### 1. Ordinary Least Squares (Base Loss)
$$\text{MSE}(w, b) = \frac{1}{N} \sum_{i=1}^N \left( y_i - (X_i \cdot w + b) \right)^2$$

### 2. Lasso Penalty ($L_1$)
$$\mathcal{L}_{\text{Lasso}} = \text{MSE}(w, b) + \alpha \sum_{j=1}^D |w_j|$$
$$\frac{\partial \mathcal{L}_{\text{Lasso}}}{\partial w_j} = \frac{\partial \text{MSE}}{\partial w_j} + \alpha \cdot \text{sign}(w_j)$$

### 3. Ridge Penalty ($L_2$)
$$\mathcal{L}_{\text{Ridge}} = \text{MSE}(w, b) + \frac{1}{2} \alpha \sum_{j=1}^D w_j^2$$
$$\frac{\partial \mathcal{L}_{\text{Ridge}}}{\partial w_j} = \frac{\partial \text{MSE}}{\partial w_j} + \alpha \cdot w_j$$

### 4. ElasticNet Penalty ($L_1 + L_2$)
$$\mathcal{L}_{\text{ElasticNet}} = \text{MSE}(w, b) + \alpha \cdot \rho \sum_{j=1}^D |w_j| + \frac{1}{2} \alpha (1 - \rho) \sum_{j=1}^D w_j^2$$
where $\rho = \text{l1\_ratio}$.

---

## Usage

### 1. Lasso Example (Feature Selection & Sparsity)

```python
import numpy as np
from rslearn.linear_model import Lasso
from rslearn.model_selection import train_test_split

# 1. Create dataset with 5 features: only the first 2 are truly predictive
np.random.seed(42)
n_samples = 150
X = np.random.randn(n_samples, 5)
# True model depends only on X[:, 0] and X[:, 1]; columns 2, 3, 4 are pure noise
y = 4.0 * X[:, 0] - 2.5 * X[:, 1] + np.random.normal(0, 0.2, size=n_samples)

# 2. Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# 3. Fit Lasso with alpha=0.1
lasso = Lasso(alpha=0.1, max_itr=5000)
lasso.fit(X_train, y_train, scale=True)

# 4. Inspect weights to observe shrinkage of irrelevant feature weights
weights, bias = lasso.get_weight_bias()
print("Lasso Trained Weights:", weights)
print("Lasso Trained Bias:   ", bias)

# 5. Evaluate
evaluations = lasso.evaluate(X=X_test, y_true=y_test)
print(f"Lasso Test R² Score: {evaluations['evaluation']['r2_score']:.4f}")
```

---

### 2. Ridge Example (Mitigating Multicollinearity)

```python
import numpy as np
from rslearn.linear_model import Ridge
from rslearn.model_selection import train_test_split

# 1. Generate collinear features (col 1 is highly correlated with col 0)
np.random.seed(42)
x0 = np.random.randn(120)
x1 = x0 + np.random.normal(0, 0.05, size=120)  # High collinearity
X = np.column_stack([x0, x1])
y = 3.0 * x0 + 1.5 * x1 + np.random.normal(0, 0.3, size=120)

# 2. Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train Ridge regressor
ridge = Ridge(alpha=0.2, max_itr=4000)
ridge.fit(X_train, y_train, scale=True)

# 4. Check stable weights and evaluate
weights, bias = ridge.get_weight_bias()
print("Ridge Weights:", weights)

eval_res = ridge.evaluate(X=X_test, y_true=y_test)
print(f"Ridge MSE: {eval_res['evaluation']['mse']:.4f}")
```

---

### 3. ElasticNet Example (Combined Penalty)

```python
import numpy as np
from rslearn.linear_model import ElasticNet
from rslearn.model_selection import train_test_split
from rslearn.loader import load_model

# 1. Generate synthetic data
np.random.seed(42)
X = np.random.uniform(10, 100, size=(160, 4))
y = 1.2 * X[:, 0] + 0.8 * X[:, 1] - 0.5 * X[:, 2] + 25.0 + np.random.normal(0, 2.0, size=160)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Train ElasticNet with balanced 50/50 L1/L2 ratio
enet = ElasticNet(alpha=0.15, l1_ratio=0.5, max_itr=5000)
enet.fit(X_train, y_train, scale=True)

# 3. Save model
enet.save("enet_model.rsl")

# 4. Restore model via loader
loaded_enet = load_model("enet_model.rslr")
preds = loaded_enet.predict(X_test)
print("First 5 ElasticNet Predictions:", preds[:5])

results = loaded_enet.evaluate(y_pred=preds, y_true=y_test)
print("ElasticNet R² Score:", results["evaluation"]["r2_score"])
```

---

# Follow up
* [LinearRegression](linear_regression.md) - Standard gradient descent regression with configurable regularization
* [LogisticRegression](logistic_regression.md) - Binary and multi-class classification
* [StandardScaler](../preprocessing/standard_scaler.md) - Feature standardization required for regularized models
* [Regression Metrics](../metrics/regression.md) - Understanding R², MSE, MAE, and RMSE
* [load_model](../loader/load_model.md) - Deserializing saved `.rslr` models
