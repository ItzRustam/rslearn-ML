# Errors

This part of the documentation contains complete information about the custom exception hierarchy in `rslearn.Errors`.

* <a href="../../README.md">Back to docs/</a>
* <a href="../README.md">Back to user_guide</a>

---

## Overview

The `Errors` module defines custom exceptions used throughout `rslearn`. These exceptions safeguard models against incorrect input shapes, mismatched sample lengths, invalid parameter arguments, and calls to unfitted estimators, providing descriptive error messages to facilitate debugging.

All custom exceptions inherit from the base `Exception` class and can be caught collectively or individually in application pipelines.

---

## Exception Hierarchy Summary

| Exception Class | Base Class | Trigger Condition | Primary Raising Modules |
|---|---|---|---|
| `Error` | `Exception` | Base application exception; raised for invalid file extensions (`.rslr`/`.rslc`) or unsupported saving types. | `rslearn.BaseEstimators`, `rslearn.loader` |
| `LengthError` | `Exception` | Raised when input array lengths do not match or empty matrices are supplied. | `rslearn.neighbors`, `rslearn.Pipeline` |
| `NotFittedError` | `Exception` | Raised when `.predict()`, `.transform()`, or `.evaluate()` is invoked before `.fit()`. | `rslearn.BaseEstimators`, `rslearn.preprocessing`, `rslearn.Pipeline`, `rslearn.neighbors` |
| `InvalidShape` | `Exception` | Raised when feature dimensions ($n_{\text{features}}$) in new test data do not match training features. | `rslearn.neighbors`, `rslearn.BaseEstimators` |
| `InvalidValueError` | `Exception` | Raised when unsupported argument values, invalid solvers, bad hyperparameter values, or unexpected `None` inputs are encountered. | `rslearn.linear_model`, `rslearn.preprocessing`, `rslearn.metrics`, `rslearn.Pipeline` |
| `InternelError` | `Exception` | Raised for internal dispatch errors, unhandled states, or corrupted files. | `rslearn.BaseEstimators`, `rslearn.loader` |

---

## Detailed Exception Reference

### 1. `NotFittedError`
Raised when attempting to generate predictions, transform data, or evaluate a model prior to training.

**Cause:** Calling `predict()`, `transform()`, or `evaluate()` on an uninitialized instance.

```python
from rslearn.preprocessing import StandardScaler
from rslearn.Errors import NotFittedError

scaler = StandardScaler()
try:
    # Attempting to transform before fitting
    scaler.transform([[10, 20]])
except NotFittedError as e:
    print(f"Caught NotFittedError: {e}")
```

---

### 2. `LengthError`
Raised when two or more arrays that must share the same number of samples differ in length, or when an empty dataset is provided.

**Cause:** Mismatched sample count between `X` and `y` (e.g. `len(X) != len(y)`).

```python
import numpy as np
from rslearn.neighbors import KNNClassifier
from rslearn.Errors import LengthError

model = KNNClassifier()
X = np.array([[1, 2], [3, 4], [5, 6]])  # 3 samples
y = np.array([0, 1])                     # 2 samples (mismatch!)

try:
    model.fit(X, y)
except LengthError as e:
    print(f"Caught LengthError: {e}")
```

---

### 3. `InvalidShape`
Raised when query feature matrices passed to `.predict()` have a different number of feature columns than the dataset used during `.fit()`.

**Cause:** Feature count mismatch (e.g. model trained on 3 features, but prediction input contains 2 features).

```python
import numpy as np
from rslearn.neighbors import KNNRegressor
from rslearn.Errors import InvalidShape

model = KNNRegressor()
X_train = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # 2 features
y_train = np.array([10.0, 20.0, 30.0])
model.fit(X_train, y_train)

X_test = np.array([[1.0, 2.0, 3.0]])  # 3 features (mismatch!)
try:
    model.predict(X_test)
except InvalidShape as e:
    print(f"Caught InvalidShape: {e}")
```

---

### 4. `InvalidValueError`
Raised when invalid parameter configurations are supplied, such as an unrecognized solver string, unsupported `multi_output` mode, negative `k_neighbors`, or `None` values where an array was required.

**Cause:** Passing unsupported parameters to estimators, scalers, or evaluation metrics.

```python
from rslearn.linear_model import LogisticRegression
from rslearn.Errors import InvalidValueError

try:
    # Invalid solver name
    model = LogisticRegression(solver="unsupported_solver")
except InvalidValueError as e:
    print(f"Caught InvalidValueError: {e}")
```

---

### 5. `Error`
Raised for file persistence violations, such as attempting to save or load a model with an unsupported extension or corrupted archive.

```python
from rslearn.loader import load_model
from rslearn.Errors import Error

try:
    # Invalid extension (must be .rslr or .rslc)
    model = load_model("model.txt")
except Error as e:
    print(f"Caught Error: {e}")
```

---

## Defensive Programming & Error Handling

When building automated production pipelines, wrapping `rslearn` workflows in standard `try...except` blocks ensures graceful error handling:

```python
import numpy as np
from rslearn.linear_model import LinearRegression
from rslearn.Errors import NotFittedError, InvalidValueError, Error

model = LinearRegression()

try:
    # Defensive execution
    predictions = model.predict(np.array([[1.0, 2.0]]))
except NotFittedError:
    print("Model not trained! Fitting model first...")
    model.fit([[1.0, 2.0], [3.0, 4.0]], [5.0, 11.0])
    predictions = model.predict(np.array([[1.0, 2.0]]))
    print("Predictions:", predictions)
```

---

# Follow up
* [user_guide Overview](../README.md) - Complete user guide architecture
* [BaseEstimators](../base_estimators/README.md) - Base class architecture and validation routines
* [loader](../loader/load_model.md) - Model loading error specifications
