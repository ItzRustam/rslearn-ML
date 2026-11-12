# Pipeline (End-to-End Workflow)

This part of the documentation contains complete information about the unified machine learning pipeline in `rslearn.Pipeline`.

* <a href="../../README.md">Back to docs/</a>
* <a href="../README.md">Back to user_guide</a>

---

## Overview

The `pipeline` class ties together preprocessing (`scaler`), model training (`model`), automatic data partitioning (`validation_split`), and evaluation reporting into a single streamlined workflow.

Instead of writing boilerplate code for scaling training data, transforming test sets, fitting estimators, generating predictions, and computing metric matrices, `pipeline` manages the entire lifecycle automatically.

---

## Key Capabilities

| Capability | Manual Approach | `rslearn.Pipeline` Approach |
|---|---|---|
| **Feature Scaling** | Fit scaler on `X_train`, transform `X_train`, transform `X_test` | Handled internally in `fit()` and `predict()` |
| **Validation Splitting** | Explicit `train_test_split` calls | Set `validation_split=True` |
| **Model Evaluation** | Compute accuracy, precision, recall, F1, or R², MSE, MAE, RMSE individually | Automatically evaluated and displayed upon fitting |
| **Inference on Raw Data** | Manually remember to call `scaler.transform(X_new)` | Call `pipeline.predict(X_new)` directly on raw features |

---

## Table Of Content
* <a href="pipeline.md">pipeline</a> - Comprehensive guide to configuring, fitting, and predicting with `rslearn.Pipeline.pipeline`
