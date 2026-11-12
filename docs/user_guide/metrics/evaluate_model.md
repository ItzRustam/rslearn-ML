# metrics.evaluate_model

## Overview
`evaluate_model` is an all-in-one evaluation function designed to compute predictions and all corresponding statistical metrics in a single line of code. It works with regression and classification models from `rslearn`, `scikit-learn`, `xgboost`, or any estimator adhering to the standard `.predict(X)` interface.

**Features**  
* Computes all essential metrics in a single unified call.
* Supports dual modes:
  1. **Model Mode**: Pass `model`, `X`, and `y_true`; it generates predictions and evaluates them.
  2. **Prediction Mode**: Pass precomputed `y_pred` and `y_true` directly without requiring a model object.
* Returns structured dictionary containing both `'prediction'` and `'evaluation'`.
* Compatible with both regression (`task="regression"`) and classification (`task="classification"`).

---

## Parameters

### Function Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | estimator object | `None` | Trained machine learning model implementing a `.predict(X)` method. |
| `X` | `array-like` of shape `(n_samples, n_features)` | `None` | Input test features. Used by `model.predict(X)` when `y_pred` is not provided. |
| `y_pred` | `array-like` of shape `(n_samples,)` | `None` | Precomputed prediction array. If provided, `model` and `X` can be omitted. |
| `y_true` | `array-like` of shape `(n_samples,)` | *Required* | Ground truth target labels or continuous values. |
| `task` | `str` | `"regression"` | Type of machine learning task: `"regression"` or `"classification"`. |

**Usage**  
```python
from rslearn.metrics import evaluate_model

# Option 1: Evaluate using model + test features
results = evaluate_model(model=model, X=X_test, y_true=y_test, task="regression")

# Option 2: Evaluate using precomputed predictions
results = evaluate_model(y_pred=y_pred, y_true=y_test, task="classification")
```

---

## Return Structure

`evaluate_model` returns a Python dictionary with two primary top-level keys:

```python
{
    "prediction": np.ndarray,  # The predicted array
    "evaluation": dict         # Dictionary of metric scores
}
```

### Metrics Returned by Task

| Task | Metrics Included in `evaluation` Dictionary |
|---|---|
| `"regression"` | `r2_score`: Coefficient of determination (float)<br>`mse`: Mean Squared Error (float)<br>`mae`: Mean Absolute Error (float)<br>`rmse`: Root Mean Squared Error (float) |
| `"classification"` | `accuracy_score`: Fraction of correct classifications (float)<br>`precision`: Precision score (float or array)<br>`recall`: Recall score (float or array)<br>`f1_score`: F1 score (float or array) |

---

## NOTE
1. **Third-Party Model Support**: `evaluate_model` works out of the box with `scikit-learn`, `LightGBM`, and `XGBoost`.
2. **Deep Learning Frameworks (PyTorch, TensorFlow, Keras)**: If your model outputs tensors, convert them to NumPy arrays before evaluating:
   ```python
   # For PyTorch:
   preds = torch_model(X_test_tensor).detach().cpu().numpy().reshape(-1)
   results = evaluate_model(y_pred=preds, y_true=y_test, task="regression")
   ```
3. **No Probabilities for Classification**: Pass discrete class label predictions, not continuous probability scores.

---

## Usage

### 1. Regression Task Evaluation

```python
import numpy as np
from rslearn.linear_model import LinearRegression
from rslearn.model_selection import train_test_split
from rslearn.metrics import evaluate_model

# Generate data
np.random.seed(42)
X = np.random.rand(100, 3)
y = 3.0 * X[:, 0] + 1.5 * X[:, 1] - 2.0 * X[:, 2] + 0.5

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# Fit model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate using model and X_test
eval_results = evaluate_model(model=model, X=X_test, y_true=y_test, task="regression")

print("Predictions:", eval_results["prediction"][:5])
print("Evaluations:", eval_results["evaluation"])
# Example output:
# {'r2_score': 0.9998, 'mse': 0.0001, 'mae': 0.0089, 'rmse': 0.0102}
```

### 2. Classification Task Evaluation

```python
import numpy as np
from rslearn.linear_model import LogisticRegression
from rslearn.model_selection import train_test_split
from rslearn.metrics import evaluate_model

# Generate classification data
np.random.seed(42)
X = np.random.randn(80, 2)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate model directly
eval_results = evaluate_model(model=model, X=X_test, y_true=y_test, task="classification")

print("Predictions:", eval_results["prediction"][:5])
print("Evaluations:", eval_results["evaluation"])
# Example output:
# {'accuracy_score': 0.9583, 'f1_score': 0.96, 'recall': 0.923, 'precision': 1.0}
```

---

# Follow up
* [Classification Metrics](classification.md) - Individual classification metrics
* [Regression Metrics](regression.md) - Individual regression metrics
* [Pipeline](../pipeline/pipeline.md) - Automate splitting, scaling, training, and evaluation
