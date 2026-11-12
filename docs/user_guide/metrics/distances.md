# metrics.EuclidienDisctance

## Overview
`EuclidienDisctance` computes the vectorized Euclidean distance between a query point (`key`) and every observation in a reference dataset (`dataset`). This function powers the nearest-neighbor search routines in `KNNClassifier` and `KNNRegressor`.

**Features**  
* High-performance vectorized calculation using NumPy's `np.linalg.norm`.
* Automatic dimension checking with clear error feedback.
* Supports arbitrary feature dimensions ($D \ge 1$).

---

## Parameters

### Function Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `dataset` | `np.ndarray` of shape `(n_samples, n_features)` | *Required* | Matrix of reference data points. |
| `key` | `np.ndarray` of shape `(n_features,)` | *Required* | Vector representing the query point to measure distance from. |

**Returns**  
* `distances`: `np.ndarray` of shape `(n_samples,)` containing the Euclidean distance from `key` to each row in `dataset`.

**Usage**  
```python
from rslearn.metrics import EuclidienDisctance

distances = EuclidienDisctance(dataset=dataset, key=key)
```

---

## Mathematical Formulation

The Euclidean distance between a key point $p = (p_1, p_2, \dots, p_D)$ and a dataset point $q = (q_1, q_2, \dots, q_D)$ in $D$-dimensional space is:

$$d(p, q) = \sqrt{\sum_{k=1}^D (p_k - q_k)^2} = \| p - q \|_2$$

---

## Usage Example

```python
import numpy as np
from rslearn.metrics import EuclidienDisctance

# Reference dataset (3 samples, 2 features)
data = np.array([
    [1.0, 2.0],
    [3.0, 4.0],
    [4.0, 5.0]
])

# Query point (2 features)
key = np.array([2.0, 3.0])

# Compute distances
distances = EuclidienDisctance(dataset=data, key=key)
print("Distances to key:", distances)
# Output: [1.41421356 1.41421356 2.82842712]

# Find nearest neighbor index
nearest_idx = np.argmin(distances)
print(f"Closest point index: {nearest_idx}, point: {data[nearest_idx]}")
```

---

# Follow up
* [KNNClassifier](../neighbors/KNNClassifier.md) - K-Nearest Neighbors classifier
* [KNNRegressor](../neighbors/KNNRegressor.md) - K-Nearest Neighbors regressor
* [metrics Overview](README.md) - All metrics in rslearn
