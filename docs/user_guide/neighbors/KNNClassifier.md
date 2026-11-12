# neighbors.KNNClassifier

## Overview
`KNNClassifier` is an instance-based supervised learning algorithm for both binary and multi-class classification tasks. It identifies the $k$ closest training examples to a query point based on Euclidean distance and assigns the most frequent class label (majority vote) among them.

**Features**  
* Supports both binary and multi-class classification targets.
* Automatic feature scaling with `StandardScaler` during `fit(scale=True)` and `predict()`.
* Manual scaling control via `hard_scale_off=True`.
* Built-in all-in-one evaluation with `evaluate()`.
* Compatible with Python lists, tuples, pandas DataFrames, and NumPy arrays.

---

## Parameters

### Class Parameter (tuning parameters)

| Parameter | Type | Default | Description |
|---|---|---|---|
| `k_neighbors` | `int` | `5` | Number of nearest neighbors to consider for classification voting. |
| `hard_scale_off` | `bool` | `True` | When `True`, internal auto-scaling is disabled and feature values are used as provided. Set to `False` to enable automatic scaling. |

**Usage**  
```python
from rslearn.neighbors import KNNClassifier

model = KNNClassifier(k_neighbors=5, hard_scale_off=False)
```

---

## NOTE
1. **Feature Scaling**: KNN distance metrics are sensitive to the relative scale of different features. If your features have not been pre-scaled using `StandardScaler` or `MinMaxScaler`, set `hard_scale_off=False` to let `KNNClassifier` scale them automatically.
2. **Tie Breaking**: If there is an equal vote among multiple classes for the $k$ neighbors, the first encountered class with the maximum count is chosen.

---

### Method Parameters

| Method | Parameters | Return Type | Description |
|---|---|---|---|
| `fit(X, y, scale=True)` | `X`: Feature matrix (`n_samples, n_features`)<br>`y`: Target labels (`n_samples,`)<br>`scale`: `bool`, default `True` | `None` | Stores training samples and computes scaling statistics if `scale=True` and `hard_scale_off=False`. |
| `predict(X_new)` | `X_new`: Feature matrix (`n_samples, n_features`) | `np.ndarray` | Computes distances to all training points, selects $k$ nearest neighbors, and returns majority-voted class labels as an `np.ndarray`. |
| `evaluate(X=None, y_pred=None, y_true=None)` | `X`: Optional feature matrix for on-the-fly predictions<br>`y_pred`: Optional precomputed predictions<br>`y_true`: Ground truth labels (`n_samples,`) | `dict` | Computes classification metrics (`accuracy_score`, `precision`, `recall`, `f1_score`) and returns a dictionary with `{"prediction": ..., "evaluation": ...}`. |

---

## Usage

### Complete End-to-End Example

```python
import numpy as np
from rslearn.neighbors import KNNClassifier
from rslearn.model_selection import train_test_split

# 1. Prepare sample classification dataset (multi-class: 0, 1, 2)
X = np.array([
    [1.0, 1.1],
    [1.2, 0.9],
    [0.8, 1.0],
    [5.0, 5.2],
    [5.1, 4.9],
    [4.8, 5.0],
    [9.0, 8.8],
    [9.2, 9.1],
    [8.9, 9.0],
    [1.1, 1.3],
    [5.3, 5.1],
    [9.1, 8.9]
])
y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 1, 2])

# 2. Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 3. Initialize and train KNNClassifier
model = KNNClassifier(k_neighbors=3, hard_scale_off=False)
model.fit(X_train, y_train, scale=True)

# 4. Generate predictions on test data
predictions = model.predict(X_test)
print(f"Predictions: {predictions}")

# 5. Evaluate the model in one call
evaluations = model.evaluate(y_pred=predictions, y_true=y_test)
print("Evaluation results:")
print(f"Accuracy:  {evaluations['evaluation']['accuracy_score']:.4f}")
print(f"Precision: {evaluations['evaluation']['precision']}")
print(f"Recall:    {evaluations['evaluation']['recall']}")
print(f"F1 Score:  {evaluations['evaluation']['f1_score']}")
```

---

# Follow up
* [KNNRegressor](KNNRegressor.md) - For continuous target variables
* [StandardScaler](../preprocessing/standard_scaler.md) - Preprocessing scaler
* [train_test_split](../model_selection/splitting.md) - Splitting dataset into train and test sets
* [Classification Metrics](../metrics/classification.md) - Accuracy, Precision, Recall, and F1
* [evaluate_model](../metrics/evaluate_model.md) - Universal evaluation function
