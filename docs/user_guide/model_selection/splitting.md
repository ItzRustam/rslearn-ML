# model_selection.train_test_split

## Overview
`train_test_split` partitions feature matrices and target arrays into separate training and testing subsets. It supports standard random shuffling as well as stratified sampling to preserve relative class distributions in imbalanced classification problems.

**Features**  
* Random dataset splitting with custom split ratios.
* Stratified sampling via `stratify=y` to maintain exact class proportions in both sets.
* Seed control via `random_state` for deterministic and reproducible splits.
* Supports diverse data structures: NumPy `ndarray`, Python `list`, `tuple`, pandas `DataFrame`, and `Series`.
* Returns clean NumPy arrays for direct input into `rslearn` estimators.

---

## Parameters

### Function Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `X` | `array-like` of shape `(n_samples, n_features)` | *Required* | Feature matrix containing sample observations. |
| `y` | `array-like` of shape `(n_samples,)` | *Required* | Target values or class labels corresponding to `X`. |
| `test_size` | `float` | `0.2` | Fraction of dataset to allocate to the test split. Must satisfy $0 < \text{test\_size} < 1$. |
| `random_state` | `int` | `None` | Integer seed controlling the random number generator for reproducible shuffles. |
| `stratify` | `array-like` of shape `(n_samples,)` | `None` | Target array to guide stratified splitting. Each class must have at least 2 samples. |

**Returns**  
* `X_train`: Training feature matrix as `np.ndarray`.
* `X_test`: Testing feature matrix as `np.ndarray`.
* `y_train`: Training target values as `np.ndarray`.
* `y_test`: Testing target values as `np.ndarray`.

**Usage**  
```python
from rslearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

---

## NOTE
1. **Sample Count Requirement**: `X` and `y` must contain the exact same number of samples (`len(X) == len(y)`), otherwise a `ValueError` is raised.
2. **Stratification Constraint**: When `stratify` is specified, each unique class must contain **at least 2 samples** to allow representation in both train and test splits.
3. **Data Types**: All outputs (`X_train`, `X_test`, `y_train`, `y_test`) are converted to `np.ndarray`.

---

## Stratified vs. Random Splitting

In classification tasks with imbalanced class distributions, simple random sampling may accidentally leave the test set with zero (or too few) minority class instances.

For example, given:
```python
y = [0, 0, 0, 0, 0, 0, 1, 1]
```
A random split without stratification might allocate all instances of class `1` into the training set, leaving the test set with zero positive examples.

By supplying `stratify=y`, `train_test_split` partitions each class independently according to `test_size`, ensuring identical class ratios in both subsets:

| Dataset | Class 0 Ratio | Class 1 Ratio |
|---|---|---|
| Original | 75% | 25% |
| Train (`stratify=y`) | 75% | 25% |
| Test (`stratify=y`) | 75% | 25% |

---

## Usage

### 1. Standard Random Split (Regression Example)

```python
import numpy as np
from rslearn.model_selection import train_test_split
from rslearn.linear_model import LinearRegression

# Generate regression features and target
X = np.array([
    [10.0, 2.0],
    [20.0, 4.0],
    [30.0, 6.0],
    [40.0, 8.0],
    [50.0, 10.0],
    [60.0, 12.0],
    [70.0, 14.0],
    [80.0, 16.0],
    [90.0, 18.0],
    [100.0, 20.0]
])
y = np.array([14.0, 28.0, 42.0, 56.0, 70.0, 84.0, 98.0, 112.0, 126.0, 140.0])

# Split 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

# Fit model
model = LinearRegression()
model.fit(X_train, y_train)
print("Predictions:", model.predict(X_test))
```

### 2. Stratified Split (Classification Example)

```python
import numpy as np
from rslearn.model_selection import train_test_split
from rslearn.linear_model import LogisticRegression

# Imbalanced classification target
X = np.random.randn(20, 3)
y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

# Perform stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print("Class distribution in Train:", np.bincount(y_train))
print("Class distribution in Test: ", np.bincount(y_test))
```

---

# Follow up
* [LogisticRegression](../linear_model/logistic_regression.md) - Classification model
* [LinearRegression](../linear_model/linear_regression.md) - Regression model
* [StandardScaler](../preprocessing/standard_scaler.md) - Feature standardization
* [evaluate_model](../metrics/evaluate_model.md) - Single-line model evaluation
