# loader

This part of the documentation contains complete information about model persistence, serialization formats, and loading saved models in `rslearn.loader`.

* <a href="../../README.md">Back to docs/</a>
* <a href="../README.md">Back to user_guide</a>

---

## Overview

`rslearn` models provide native serialization via `.save()` and deserialization via `load_model()`. Models are saved as lightweight, gzip-compressed JSON files storing:
1. Model parameters (weights, bias, hyperparameters, and solver configurations).
2. Internal preprocessor statistics (e.g. `StandardScaler` mean and standard deviation) to ensure test predictions use the exact scaling distribution learned during training.

---

## Supported Formats

| Format Extension | Target Model Family | Task | Contents Saved |
|---|---|---|---|
| `.rslr` | `LinearRegression`, `Ridge`, `Lasso`, `ElasticNet` | Regression | Weights, bias, alpha, l1_ratio, max_itr, scaler statistics |
| `.rslc` | `LogisticRegression` (`liblinear` & `saga`) | Classification | Solver, class weights/biases, categorical submodels, scaler statistics |

---

## Table Of Content
* <a href="load_model.md">Load Model</a> - Restore pre-trained models from `.rslr` and `.rslc` files
