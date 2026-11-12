# Docs/
> Comprehensive documentation for `rslearn-ML` for library users and developers.

## Overview
`rslearn-ML` is an educational, pure NumPy machine learning library featuring gradient descent linear models, K-Nearest Neighbors, automated end-to-end pipelines, comprehensive statistical metrics, and compressed model serialization.

This documentation provides practical "how-to" tutorials, exhaustive class and method parameter specifications, mathematical formulations, and architectural diagrams.

---

## Setup & Installation

Before exploring the documentation or running example scripts, set up your Python environment:

```bash
# Standard installation from PyPI
pip install rslearn-py==1.0.4

# Or install latest release
pip install --upgrade rslearn-py
```

### Preferred Environment
* Python 3.8+
* Virtual Environment (`python -m venv venv`)
* Core Dependencies: `numpy`, `pandas`, `kagglehub`

---

## Documentation Map

| Guide Section | Target Audience | Primary Focus | Link |
|---|---|---|---|
| **Quick Start Guide** | Beginners & Practical Users | Fast, runnable code snippets covering all major tasks | <a href="quick_start.md">`quick_start.md`</a> |
| **User Guide (Full)** | Library Users | In-depth API reference, parameters, tables, and mathematics | <a href="user_guide/README.md">`user_guide/README.md`</a> |
| **Contributor Guide** | Maintainers & Developers | Repository architecture, coding conventions, extending modules | <a href="contributor_guide/README.md">`contributor_guide/README.md`</a> |

---

## User Guide Modules

All subdirectories contain their own `README.md` serving as an introductory chapter guide:

| Module | Core Purpose | Key Algorithms & Classes | Link |
|---|---|---|---|
| **`linear_model`** | Gradient-descent linear estimators & regularizers | `LinearRegression`, `LogisticRegression`, `Ridge`, `Lasso`, `ElasticNet` | <a href="user_guide/linear_model/README.md">`user_guide/linear_model/`</a> |
| **`neighbors`** | Memory-based instance learning | `KNNClassifier`, `KNNRegressor` | <a href="user_guide/neighbors/README.md">`user_guide/neighbors/`</a> |
| **`Pipeline`** | Unified end-to-end ML workflows | `pipeline` | <a href="user_guide/pipeline/README.md">`user_guide/pipeline/`</a> |
| **`preprocessing`** | Feature scaling and standardization | `StandardScaler`, `MinMaxScaler` | <a href="user_guide/preprocessing/README.md">`user_guide/preprocessing/`</a> |
| **`model_selection`** | Dataset splitting & parameter tuning | `train_test_split`, `GridSearchCV` | <a href="user_guide/model_selection/README.md">`user_guide/model_selection/`</a> |
| **`metrics`** | Performance evaluation & loss functions | `r2_score`, `mse`, `mae`, `rmse`, `accuracy_score`, `confusion_metrics`, `precision`, `recall`, `f1_score`, `evaluate_model`, `EuclidienDisctance` | <a href="user_guide/metrics/README.md">`user_guide/metrics/`</a> |
| **`loader`** | Model deserialization & deployment | `load_model` (`.rslr`, `.rslc`) | <a href="user_guide/loader/README.md">`user_guide/loader/`</a> |
| **`Errors`** | Application exception hierarchy | `Error`, `LengthError`, `NotFittedError`, `InvalidShape`, `InvalidValueError`, `InternelError` | <a href="user_guide/errors/README.md">`user_guide/errors/`</a> |
| **`BaseEstimators`** | OOP architecture & developer mixins | `BaseEstimator`, `BaseEstimatorKNN`, `BaseEstimatorRegulization`, `backupScaler` | <a href="user_guide/base_estimators/README.md">`user_guide/base_estimators/`</a> |

---

## Contributor Guide

For developers looking to contribute new algorithms, optimize NumPy routines, or maintain test suites:

* <a href="contributor_guide/README.md">Contributor Guide (Full)</a>
  * Repository Architecture & Source Map
  * Module Responsibility Matrix
  * Extending Estimators, Metrics, and Scalers
  * Testing Guidelines and Verification Suites
  * Pull Request Checklist
