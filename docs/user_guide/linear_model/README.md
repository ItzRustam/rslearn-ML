# linear_model

This part of the documentation contains complete information about gradient descent-based linear models and regularization techniques in `rslearn.linear_model`.

* <a href="../../README.md">Back to docs/</a>
* <a href="../README.md">Back to user_guide</a>

---

## Overview

The `linear_model` module implements foundational supervised machine learning algorithms powered by iterative Gradient Descent optimization. It provides solutions for continuous target estimation (Regression) and categorical class prediction (Classification), with native support for feature scaling, multi-metric performance evaluation, and model serialization.

Key highlights include:
* **Gradient Descent Optimization**: All models iteratively learn optimal weights and biases to minimize task-specific loss functions.
* **Integrated Automatic Scaling**: Seamlessly protects gradient stability by automatically scaling features via `StandardScaler` (`scale=True`) or `backupScaler` (`scale=False`).
* **Built-in Overfitting Penalties**: Flexible $L_1$ (Lasso), $L_2$ (Ridge), and ElasticNet regularization directly embedded or available via standalone classes.
* **All-in-One Evaluation**: Unified `.evaluate()` method computes relevant metric suites in a single line.
* **Serialization & Persistence**: Native `.save()` export to gzip-compressed `.rslr` (regression) and `.rslc` (classification) files preserving both parameters and scaler statistics.

---

## Architecture Summary

| Estimator | Type | Primary Task | Optimization / Loss | Regularization | Supported Formats |
|---|---|---|---|---|---|
| [`LinearRegression`](linear_regression.md) | Class | Continuous Regression | Mean Squared Error (MSE) Gradient Descent | None, L1, L2, ElasticNet | `.rslr` |
| [`LogisticRegression`](logistic_regression.md) | Class | Binary & Multi-Class Classification | Binary Cross-Entropy / OvR Log Loss | Gradient Descent with class weighting | `.rslc` |
| [`Lasso`](regulizations.md#lasso-l1-regularization) | Class | Continuous Regression | MSE + L1 norm penalty ($|w|$) | L1 penalty | `.rslr` |
| [`Ridge`](regulizations.md#ridge-l2-regularization) | Class | Continuous Regression | MSE + L2 norm penalty ($w^2$) | L2 penalty | `.rslr` |
| [`ElasticNet`](regulizations.md#elasticnet-l1--l2-regularization) | Class | Continuous Regression | MSE + convex combination of L1 and L2 | L1 + L2 penalties | `.rslr` |

---

## Solver & Algorithm Guide

### Regression Estimators
For continuous target values ($y \in \mathbb{R}$):
* Use **`LinearRegression`** for standard linear relationships with optional regularization.
* Use **`Lasso`** when feature selection is required (drives non-informative feature weights to zero).
* Use **`Ridge`** when features are highly collinear or multicollinearity is suspected.
* Use **`ElasticNet`** when dealing with high-dimensional data with groups of correlated features.

### Classification Estimators
For discrete target classes ($y \in \{0, 1, \dots, C-1\}$):
* **`LogisticRegression(solver="liblinear")`**: Tailored for binary classification tasks ($C = 2$).
* **`LogisticRegression(solver="saga")`**: Employs One-vs-Rest (OvR) multi-model ensembles for multi-class classification ($C > 2$).
* **`LogisticRegression(solver="auto")`**: Automatically inspects target $y$ during `fit()` and selects `liblinear` for binary targets or `saga` for multi-class targets.

---

## Feature Scaling in Linear Models

Gradient descent updates parameters iteratively:
$$w \leftarrow w - \eta \cdot \frac{\partial \mathcal{L}}{\partial w}$$

When features have disparate scales, gradient updates oscillate wildly along high-variance dimensions, causing slow convergence or numerical instability.

| Configuration | Feature Treatment | Recommendation |
|---|---|---|
| `scale=True` (default), `hard_scale_off=False` | Automatically fits and transforms data with internal `StandardScaler`. | **Recommended for most workflows.** |
| `scale=False`, `hard_scale_off=False` | Applies secondary `backupScaler` ($X / \max(X)$) to prevent gradient explosion. | Use when basic normalization is desired without zero-centering. |
| `hard_scale_off=True` | Completely disables all internal scaling routines; uses raw features directly. | Use **only** when data has already been scaled (e.g. by `StandardScaler` or in a `pipeline`). |

---

## Table Of Content

* <a href="linear_regression.md">LinearRegression</a> - Gradient descent-based linear regression with integrated regularization and scaling
* <a href="logistic_regression.md">LogisticRegression</a> - Binary and multi-class classification with automated solver selection
* <a href="regulizations.md">Regularizations (Lasso, Ridge, ElasticNet)</a> - Standalone regularized estimators to prevent overfitting
