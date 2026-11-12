# Contributor Guide: rslearn-ML

> Comprehensive developer and contributor guide for maintaining and extending `rslearn-ML`.

* <a href="../README.md">Back to docs/</a>
* <a href="../user_guide/README.md">User Guide</a>

---

## Overview

`rslearn-ML` is a lightweight, high-performance machine learning library implemented in pure Python and NumPy. The library prioritizes clarity, educational accessibility, and practical utility, featuring built-in feature scaling, multi-metric performance evaluations, and safe JSON/gzip model serialization.

This guide outlines the architectural philosophy, directory layout, code conventions, extension recipes, and pull request workflows required for contributing to `rslearn`.

---

## Repository Architecture & Source Map

```
rslearn-ML/
├── rslearn/                      # Core Python package root
│   ├── __init__.py               # Top-level exports and package metadata
│   ├── version.py                # __version__ and __github_version__
│   ├── BaseEstimators/           # Abstract base classes and array utilities
│   │   ├── _base.py              # Array converters, shape checkers, multi-output logic
│   │   └── _estimator.py         # BaseEstimator, backupScaler, BaseEstimatorKNN
│   ├── linear_model/             # Gradient-descent linear algorithms
│   │   ├── _LinearRegression.py  # LinearRegression with L1/L2/ElasticNet penalties
│   │   ├── _LogisticRegression.py# Binary (liblinear) and multi-class (saga) logistic
│   │   └── _regulizations.py     # Standalone Lasso, Ridge, and ElasticNet wrappers
│   ├── neighbors/                # Instance-based nearest neighbor algorithms
│   │   ├── _knnClass.py          # KNNClassifier (majority voting)
│   │   └── _knnReg.py            # KNNRegressor (mean neighbor aggregation)
│   ├── Pipeline/                 # End-to-end automated ML pipeline
│   │   └── _pipeline.py          # pipeline class managing scaling, splitting, training
│   ├── preprocessing/            # Data standardization and normalization
│   │   └── _scaler.py            # StandardScaler and MinMaxScaler
│   ├── model_selection/          # Splitting and hyperparameter tuning
│   │   ├── _split.py             # train_test_split with stratified sampling
│   │   └── _cvs.py               # GridSearchCV (hyperparameter search)
│   ├── metrics/                  # Evaluation metrics and loss functions
│   │   ├── _classification.py    # accuracy, precision, recall, f1, confusion_matrix
│   │   ├── _regression.py        # r2_score, mse, mae, rmse (with multi-output support)
│   │   ├── _evaluate.py          # Universal evaluate_model function
│   │   └── _distances.py         # Vectorized EuclidienDisctance
│   ├── loader/                   # Model deserialization and restoration
│   │   └── _model_loader.py      # load_model (.rslr and .rslc parser)
│   └── Errors/                   # Custom application exceptions
│       └── _errors.py            # NotFittedError, InvalidShape, LengthError, etc.
├── docs/                         # Markdown documentation portal
├── tests/                        # Verification scripts and test suites
└── pyproject.toml                # Build configuration and dependency specification
```

---

## Module Responsibility Matrix

| Module | Purpose | Public Interface Classes & Functions |
|---|---|---|
| `rslearn.linear_model` | Gradient-descent supervised algorithms | `LinearRegression`, `LogisticRegression`, `Ridge`, `Lasso`, `ElasticNet` |
| `rslearn.neighbors` | Memory-based nearest neighbors | `KNNClassifier`, `KNNRegressor` |
| `rslearn.Pipeline` | Automated scaling, splitting, and training | `pipeline` |
| `rslearn.preprocessing` | Feature standardization | `StandardScaler`, `MinMaxScaler` |
| `rslearn.model_selection` | Dataset partitioning and tuning | `train_test_split`, `GridSearchCV` |
| `rslearn.metrics` | Task performance measurement | `r2_score`, `mse`, `mae`, `rmse`, `accuracy_score`, `confusion_metrics`, `precision`, `recall`, `f1_score`, `evaluate_model`, `EuclidienDisctance` |
| `rslearn.loader` | Compressed model deserialization | `load_model` |
| `rslearn.Errors` | Exception handling hierarchy | `Error`, `LengthError`, `NotFittedError`, `InvalidShape`, `InvalidValueError`, `InternelError` |
| `rslearn.BaseEstimators` | Object-oriented base infrastructure | `BaseEstimator`, `BaseEstimatorKNN`, `BaseEstimatorRegulization`, `backupScaler` |

---

## Core Architectural Principles

When writing or modifying code in `rslearn`, adhere to the following principles:

1. **Pure NumPy Vectorization**:
   All mathematical operations (gradients, distances, metrics, loss functions) must be vectorized using NumPy arrays. Never use explicit Python `for` loops across rows when vector operations (`np.dot`, `@`, broadcasting, `np.linalg.norm`) are applicable.
2. **Inheritance from BaseEstimator**:
   All new estimators must subclass `BaseEstimator` (for parametric models) or `BaseEstimatorKNN` (for instance models). This provides automatic integration with `_scale_True`, `_scale_False`, `_eval`, and `.save()`.
3. **Integrated Scaling**:
   Estimators must support `scale=True` (StandardScaler) and `hard_scale_off=True` (bypass scaling) to ensure consistent user experience across the library.
4. **Safety & Robustness**:
   Use `rslearn.BaseEstimators._base` helper routines (`convert_array`, `convert1D`, `shape_checker`) and raise typed exceptions from `rslearn.Errors`. Never fail with generic `IndexError` or unhandled `KeyError`.
5. **No Dangerous Serialization**:
   Never use Python's `pickle` for model saving. Models must serialize as gzip-compressed JSON storing parameters, hyperparameters, and scaler distribution statistics.

---

## Extending the Library: Developer Recipes

### 1. Adding a New Estimator
To create a new estimator, inherit from `BaseEstimator`:

```python
import numpy as np
from rslearn.BaseEstimators import BaseEstimator
from rslearn.BaseEstimators import _base
from rslearn.Errors import NotFittedError, InvalidValueError

class MyNewRegressor(BaseEstimator):
    def __init__(self, lr=0.01, max_itr=1000, hard_scale_off=False):
        super().__init__(lr=lr, max_itr=max_itr)
        self.type = "regression"      # "regression" or "classification"
        self._model = "MyNewRegressor"
        self.hard_scale_off = hard_scale_off

    def fit(self, X, y, scale=True):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        _base.shape_checker(X, y, output_mode=False)

        # Handle automatic scaling
        if self.hard_scale_off:
            pass
        elif scale:
            X = self._scale_True(X, scaled=False)
            self.flag = True
        else:
            X = self._scale_False(X, scaled=False)
            self.flag = False

        self.fitted_shape = X.shape
        # ... Perform training / gradient optimization ...
        self._fitted = True

    def predict(self, X):
        if not self._fitted:
            raise NotFittedError("Model not fitted yet.")
        X = np.asarray(X, dtype=float)
        if self.flag:
            X = self._scale_True(X, scaled=True)
        # ... Compute predictions ...
        return predictions

    def evaluate(self, X=None, y_pred=None, y_true=None):
        return super()._eval(X=X, y_pred=y_pred, y_true=y_true)
```

---

### 2. Adding a New Evaluation Metric
All regression and classification metrics in `rslearn.metrics` must support:
* 1D inputs (`(n_samples,)`)
* 2D multi-output matrices (`(n_samples, n_outputs)`)
* The standard `multi_output` modes: `"uniform_average"`, `"weighted"`, and `"raw_values"`.

```python
import numpy as np
from rslearn.BaseEstimators._base import (
    convert_array,
    check_multioutput,
    dim_validator,
    shape_checker,
    multi_output_selector
)

def my_new_metric(y_true, y_pred, multi_output="uniform_average", weights=None):
    check_multioutput(multi_output)
    y_true, y_pred = convert_array(y_true, y_pred)
    shape_checker(y_true, y_pred, output_mode=True)

    if dim_validator(y_true):
        # 1D calculation
        score = np.mean(np.abs(y_true - y_pred))
        return float(score)

    # Multi-output 2D calculation (column-wise)
    scores = np.mean(np.abs(y_true - y_pred), axis=0)
    return multi_output_selector(multi_output, scores, weights=weights)
```

---

### 3. Adding a New Preprocessor
All scalers in `rslearn.preprocessing` must implement:
* `fit(X)`: Computes and stores distribution parameters.
* `transform(X)`: Applies transformation and returns a new `np.ndarray`.
* `fit_transform(X)`: Performs fit and transform in one step.
* `inverse_transform(X)`: Reverts scaled data back to original coordinate space.

---

## Testing Guidelines

Run verification tests locally before submitting changes:

```bash
# Test regression algorithms and loaders
python tests/regression_testings.py

# Test classification algorithms and pipelines
python tests/classification_testings.py
```

### Writing New Unit Tests
Place tests under `tests/` verifying:
1. Convergence of gradient descent on synthetic linear/logistic data.
2. Equality between `fit_transform(X)` and `fit(X).transform(X)`.
3. Reversibility of `inverse_transform(transform(X))`.
4. Exception triggers for `NotFittedError`, `LengthError`, `InvalidShape`, and `InvalidValueError`.
5. Persistence roundtrips: `model.save()` $\rightarrow$ `load_model()` $\rightarrow$ prediction parity.

---

## Contribution Checklist

Before submitting a Pull Request, verify:

| Step | Requirement | Verified? |
|---|---|---|
| **Code Style** | Pure Python & NumPy; PEP 8 naming conventions. | [ ] |
| **Type Validation** | Validated using `rslearn.BaseEstimators._base` routines. | [ ] |
| **Error Handling** | Throws appropriate exceptions from `rslearn.Errors`. | [ ] |
| **Documentation** | Documentation added or updated in `docs/` following the table/heading schema. | [ ] |
| **Testing** | Existing tests pass and new unit tests are included. | [ ] |
| **Persistence** | Model saving/loading verified if modifying estimators. | [ ] |

---

# Follow up
* [User Guide Overview](../user_guide/README.md) - Complete documentation for library users
* [BaseEstimators](../user_guide/base_estimators/README.md) - Deep dive into BaseEstimators architecture
* [Errors Hierarchy](../user_guide/errors/README.md) - Reference for library exception classes
