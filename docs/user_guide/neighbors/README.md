# neighbors (KNN Family)

This part of the documentation contains complete information about K-Nearest Neighbors algorithms in `rslearn`, including `KNNClassifier` for classification tasks and `KNNRegressor` for continuous regression tasks.

* <a href="../../README.md">Back to docs/</a>
* <a href="../README.md">Back to user_guide</a>

---

## Overview

The K-Nearest Neighbors (KNN) family is a set of non-parametric, instance-based supervised learning algorithms. Instead of learning an explicit weight vector or hyperplane during training, KNN stores the training dataset and performs inference by calculating distances to existing data points.

In `rslearn`, KNN models include:
* Built-in Euclidean distance computation via NumPy vectorization.
* Integrated automated feature scaling via `StandardScaler` (`scale=True`) or `backupScaler`.
* Full support for user-controlled scaling bypass via `hard_scale_off`.
* Seamless multi-metric evaluation using `model.evaluate()`.

---

## Note: Feature Scaling with KNN
Distance-based algorithms like KNN calculate distances between data points using all feature dimensions. When features are on vastly different scales (for example, income in thousands vs. age in tens), features with larger magnitudes dominate the distance calculations.
Therefore, feature scaling is essential. By default, `rslearn` models automatically scale features during `fit()` and `predict()` unless `hard_scale_off=True`.

---

## Available Models

| Model | Task Type | Prediction Strategy | Default Metrics Evaluated |
|---|---|---|---|
| [`KNNClassifier`](KNNClassifier.md) | Classification (Binary & Multi-class) | Majority voting among $k$ nearest neighbors | `accuracy_score`, `precision`, `recall`, `f1_score` |
| [`KNNRegressor`](KNNRegressor.md) | Regression (Continuous) | Arithmetic mean of target values of $k$ nearest neighbors | `r2_score`, `mse`, `mae`, `rmse` |

---

## Table Of Content
* <a href="KNNClassifier.md">KNNClassifier</a> - K-Nearest Neighbors for binary and multi-class classification
* <a href="KNNRegressor.md">KNNRegressor</a> - K-Nearest Neighbors for continuous regression
