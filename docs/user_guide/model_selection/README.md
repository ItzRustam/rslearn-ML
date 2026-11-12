# model_selection

This part of the documentation contains complete information about dataset partitioning and hyperparameter optimization utilities in `rslearn.model_selection`.

* <a href="../../README.md">Back to docs/</a>
* <a href="../README.md">Back to user_guide</a>

---

## Overview

Proper model evaluation requires isolating unseen test data from the training process to accurately measure generalization performance and detect overfitting. The `model_selection` module provides tools to split data and tune model parameters.

Key capabilities include:
* Random splitting with reproducibility control via `random_state`.
* Stratified splitting (`stratify`) to preserve exact class proportion distributions in classification tasks.
* Full compatibility with NumPy arrays, lists, tuples, and pandas DataFrames.
* Automated hyperparameter tuning and model evaluation through parameter grid search.

---

## Available Tools

| Tool | Type | Key Parameters | Purpose |
|---|---|---|---|
| [`train_test_split`](splitting.md) | Function | `X`, `y`, `test_size=0.2`, `random_state=None`, `stratify=None` | Splits arrays or matrices into random train and test subsets with optional stratification. |
| [`GridSearchCV`](grid_search_cv.md) | Class (Beta) | `model`, `scoring=None`, `params={}`, `cv=2` | Systematic hyperparameter search to determine optimal estimator configurations. |

---

## Splitting Strategy Guide

| Splitting Method | Parameter Configuration | When to Use | Effect on Output |
|---|---|---|---|
| **Random Split** | `stratify=None` | Continuous regression, balanced classification datasets | Uniformly random shuffle and partition |
| **Stratified Split** | `stratify=y` | Classification with imbalanced classes ($y$) | Enforces identical class ratios in both `y_train` and `y_test` |

---

## Table Of Content
* <a href="splitting.md">train_test_split</a> - Partition datasets into train and test splits with optional stratification
* <a href="grid_search_cv.md">GridSearchCV</a> - Exhaustive hyperparameter grid search and model selection
