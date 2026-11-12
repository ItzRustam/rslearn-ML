# metrics (Model Evaluation)

This part of the documentation contains complete information about statistical evaluation metrics and automated model diagnostic utilities in `rslearn.metrics`.

* <a href="../../README.md">Back to docs/</a>
* <a href="../README.md">Back to user_guide</a>

---

## Overview

The `metrics` module provides from-scratch NumPy implementations of core evaluation functions for both regression and classification models. It supports 1D arrays as well as multi-output matrices, with uniform averaging, weighted aggregation, and raw-value outputs.

In addition to individual metric functions, `rslearn` provides `evaluate_model()`, a single-line function that generates predictions and computes all relevant metrics simultaneously.

---

## Metric Suites Overview

### 1. Classification Metrics

| Function | Output Range | Description |
|---|---|---|
| [`accuracy_score`](classification.md#accuracy_score) | $[0.0, 1.0]$ or count | Fraction or count of correctly classified samples. |
| [`confusion_metrics`](classification.md#confusion_metrics) | Matrix $(C \times C)$ | Contingency table mapping true vs. predicted class labels. |
| [`precision`](classification.md#precision) | $[0.0, 1.0]$ | Proportion of predicted positive identifications that were correct ($TP / (TP + FP)$). |
| [`recall`](classification.md#recall) | $[0.0, 1.0]$ | Proportion of actual positives identified correctly ($TP / (TP + FN)$). |
| [`f1_score`](classification.md#f1_score) | $[0.0, 1.0]$ | Harmonic mean of precision and recall ($2 \cdot \frac{P \cdot R}{P + R}$). |

### 2. Regression Metrics

| Function | Mathematical Formulation | Best Score | Description |
|---|---|---|---|
| [`r2_score`](regression.md#r2_score) | $1 - \frac{\sum (y - \hat{y})^2}{\sum (y - \bar{y})^2}$ | $1.0$ | Coefficient of determination; measures variance explained by model. |
| [`mse`](regression.md#mse) | $\frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2$ | $0.0$ | Mean Squared Error; penalizes large errors heavily. |
| [`mae`](regression.md#mae) | $\frac{1}{N} \sum_{i=1}^N \|y_i - \hat{y}_i\|$ | $0.0$ | Mean Absolute Error; robust linear penalty for errors. |
| [`rmse`](regression.md#rmse) | $\sqrt{\text{MSE}}$ | $0.0$ | Root Mean Squared Error; error metric in target variable's original scale. |

### 3. High-Level Evaluation & Utilities

| Function | Description |
|---|---|
| [`evaluate_model`](evaluate_model.md) | Universal one-line model evaluator returning both predictions and comprehensive metric scores. |
| [`EuclidienDisctance`](distances.md) | Vectorized Euclidean distance calculation between a query key and a matrix of data points. |

---

## Table Of Content
* <a href="classification.md">Classification Metrics</a> - Accuracy, Confusion Matrix, Precision, Recall, and F1 Score
* <a href="regression.md">Regression Metrics</a> - R² Score, MSE, MAE, and RMSE
* <a href="evaluate_model.md">evaluate_model</a> - Universal all-in-one evaluation tool
* <a href="distances.md">Distance Metrics</a> - Vectorized Euclidean distance computation
