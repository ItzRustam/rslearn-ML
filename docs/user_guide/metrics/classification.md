# metrics.classification

## Overview
The `classification` module within `rslearn.metrics` implements foundational evaluation metrics for binary, multi-class, and multi-output classification problems. All functions are built with pure NumPy vectorization and provide compatibility with scikit-learn conventions.

**Features**  
* Supports both binary and multi-class classification.
* Supports multi-output classification matrices.
* Configurable output aggregation (`uniform_average`, `weighted`, `raw_values`).
* Automatic dimensionality reshaping (e.g., `(n, 1)` to `(n,)`).
* Accepts Python lists, tuples, and NumPy arrays.

---

## Common Parameters

Most classification metrics accept the following standard arguments:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `y_true` | `array-like` of shape `(n_samples,)` or `(n_samples, n_outputs)` | *Required* | Ground truth (correct) target labels. |
| `y_pred` | `array-like` of shape `(n_samples,)` or `(n_samples, n_outputs)` | *Required* | Estimated target labels returned by a classifier. |

---

## Metric Functions

### 1. `accuracy_score`

Calculates the proportion or raw count of correct classifications.

#### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `y_true` | `array-like` | *Required* | Ground truth target labels. |
| `y_pred` | `array-like` | *Required* | Predicted class labels. |
| `weights` | `array-like` of shape `(n_samples,)` | `None` | Sample weights for computing weighted accuracy. |
| `multi_output` | `str` | `"uniform_average"` | Aggregation strategy: `"uniform_average"`, `"weighted"`, or `"raw_values"`. |
| `normalize` | `bool` | `True` | If `True`, returns the fraction of correct predictions ($[0.0, 1.0]$). If `False`, returns the integer count of correct predictions. |

#### Usage
```python
from rslearn.metrics import accuracy_score

y_true = [0, 1, 2, 2, 1]
y_pred = [0, 1, 2, 1, 1]

# Normalized accuracy (fraction)
score = accuracy_score(y_true, y_pred, normalize=True)
print(f"Accuracy: {score:.2f}")  # 0.80

# Raw count of correct samples
count = accuracy_score(y_true, y_pred, normalize=False)
print(f"Correct Count: {count}")  # 4.0
```

---

### 2. `confusion_metrics`

Computes the confusion matrix of size $(C \times C)$ where $C$ is the number of unique classes. Matrix row $i$ represents true instances of class $i$, while column $j$ represents predictions classified as class $j$.

#### Returns
`np.ndarray` of shape `(n_classes, n_classes)`.

#### Usage
```python
from rslearn.metrics import confusion_metrics

y_true = [0, 1, 0, 1, 1, 0]
y_pred = [0, 1, 0, 0, 1, 1]

cm = confusion_metrics(y_true, y_pred)
print("Confusion Matrix:")
print(cm)
```

---

### 3. `precision`

Computes the precision score, measuring the classifier's ability not to label a negative sample as positive:
$$\text{Precision} = \frac{TP}{TP + FP}$$

#### Usage
```python
from rslearn.metrics import precision

y_true = [1, 0, 1, 1, 0, 1]
y_pred = [1, 1, 1, 0, 0, 1]

prec = precision(y_true, y_pred)
print(f"Precision: {prec:.4f}")
```

---

### 4. `recall`

Computes the recall (sensitivity), measuring the classifier's ability to find all positive samples:
$$\text{Recall} = \frac{TP}{TP + FN}$$

#### Usage
```python
from rslearn.metrics import recall

y_true = [1, 0, 1, 1, 0, 1]
y_pred = [1, 1, 1, 0, 0, 1]

rec = recall(y_true, y_pred)
print(f"Recall: {rec:.4f}")
```

---

### 5. `f1_score`

Calculates the harmonic mean of precision and recall:
$$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

#### Usage
```python
from rslearn.metrics import f1_score

y_true = [1, 0, 1, 1, 0, 1]
y_pred = [1, 1, 1, 0, 0, 1]

f1 = f1_score(y_true, y_pred)
print(f"F1 Score: {f1:.4f}")
```

---

## Multi-Output Handling

When working with multi-output predictions (where each sample has multiple target labels), use the `multi_output` argument:

| Option | Behavior |
|---|---|
| `"uniform_average"` | Computes score for each output column and returns unweighted mean across outputs (default). |
| `"weighted"` | Computes score per output and takes weighted average using `weights`. |
| `"raw_values"` | Returns an array of scores, one for each output dimension. |

```python
import numpy as np
from rslearn.metrics import accuracy_score

y_true = np.array([[1, 0], [1, 1], [0, 1]])
y_pred = np.array([[1, 0], [1, 0], [0, 1]])

# Average across outputs
print(accuracy_score(y_true, y_pred, multi_output="uniform_average"))

# Individual column scores
print(accuracy_score(y_true, y_pred, multi_output="raw_values"))
```

---

## Complete Practical Example

```python
import numpy as np
from rslearn.metrics import (
    accuracy_score,
    confusion_metrics,
    precision,
    recall,
    f1_score
)

# Simulated predictions and ground truth
y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
y_pred = np.array([0, 2, 2, 0, 1, 1, 0, 1, 2])

# Compute all classification metrics
acc = accuracy_score(y_true, y_pred)
cm  = confusion_metrics(y_true, y_pred)
pr  = precision(y_true, y_pred)
rc  = recall(y_true, y_pred)
f1  = f1_score(y_true, y_pred)

print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {pr:.4f}")
print(f"Recall:    {rc:.4f}")
print(f"F1 Score:  {f1:.4f}")
print("\nConfusion Matrix:\n", cm)
```

---

# Follow up
* [Regression Metrics](regression.md) - R², MSE, MAE, and RMSE
* [evaluate_model](evaluate_model.md) - Single function to evaluate all metrics
* [LogisticRegression](../linear_model/logistic_regression.md) - Linear classification estimator
* [KNNClassifier](../neighbors/KNNClassifier.md) - K-Nearest Neighbors classifier
