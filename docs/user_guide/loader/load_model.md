# loader.load_model

## Overview
`load_model` restores previously trained and saved `rslearn` estimators from disk. It uncompresses gzip archives, deserializes model weights, biases, hyperparameters, and crucially **restores the internal feature scaler statistics** so new predictions can be generated immediately without retraining or manual re-scaling.

**Features**  
* Supports both regression (`.rslr`) and classification (`.rslc`) models.
* Restores internal `StandardScaler` or `backupScaler` states automatically.
* Safe and portable: uses gzip-compressed JSON instead of unsafe Python `pickle` objects.
* Restored models are ready for `.predict()` and `.evaluate()` immediately.

---

## Parameters

### Function Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `file_path` | `str` | `"rslearn_model.rsl"` | Relative or absolute path to the saved model file. Must end with `.rslr` or `.rslc`. |

**Returns**  
* Restored estimator instance (`LinearRegression` or `LogisticRegression`) marked with `_fitted=True` and ready for inference.

**Usage**  
```python
from rslearn.loader import load_model

model = load_model("my_model.rslr")
predictions = model.predict(X_new)
```

---

## Supported Model Formats

| Extension | Model Class | Saved Information |
|---|---|---|
| `.rslr` | `LinearRegression`, `Ridge`, `Lasso`, `ElasticNet` | Weight vector, bias, regularization type, alpha, l1_ratio, max_itr, learning rate, and scaler mean/std statistics. |
| `.rslc` | `LogisticRegression` | Solver (`liblinear` or `saga`), binary weights/bias or categorical multi-class submodels, learning rate, max_itr, and scaler statistics. |

---

## NOTE
1. **File Extension Enforcement**: The file must have a `.rslr` extension for regression models or `.rslc` for classification models. Supplying any other extension raises an `Error`.
2. **Scaler State Preservation**: If the model was fitted with `scale=True`, the exact feature mean and standard deviation learned from the training data are preserved inside the file. When predicting on new data with the loaded model, `rslearn` uses those exact historical scaling parameters, preventing distribution shifts.
3. **No Retraining Required**: Restored models have `_fitted = True` and can immediately call `predict(X_new)`.

---

## Usage

### 1. Saving and Loading a Linear Regression Model (`.rslr`)

```python
import numpy as np
from rslearn.linear_model import LinearRegression
from rslearn.loader import load_model

# 1. Train model
X_train = np.array([[10, 2], [20, 4], [30, 6], [40, 8]])
y_train = np.array([24, 44, 64, 84])

model = LinearRegression(regulization="l2", alpha=0.1)
model.fit(X_train, y_train, scale=True)

# 2. Save trained model
model.save("housing_model.rslr")
print("Model saved successfully as housing_model.rslr")

# 3. Load model in a new session or deployment script
loaded_model = load_model("housing_model.rslr")

# 4. Generate predictions directly on unscaled raw features
X_new = np.array([[50, 10], [60, 12]])
preds = loaded_model.predict(X_new)
print("Loaded model predictions:", preds)
```

### 2. Saving and Loading a Logistic Regression Model (`.rslc`)

```python
import numpy as np
from rslearn.linear_model import LogisticRegression
from rslearn.loader import load_model

# 1. Train multi-class model
X_train = np.array([[1.0, 1.2], [1.1, 0.9], [5.0, 5.2], [5.1, 4.8], [9.0, 8.8], [9.2, 9.1]])
y_train = np.array([0, 0, 1, 1, 2, 2])

clf = LogisticRegression(solver="auto")
clf.fit(X_train, y_train, scale=True)

# 2. Save model
clf.save("classifier.rslc")

# 3. Load model
loaded_clf = load_model("classifier.rslc")

# 4. Predict new classes
X_test = np.array([[1.0, 1.0], [5.2, 5.0], [9.1, 9.0]])
print("Loaded classifier predictions:", loaded_clf.predict(X_test))
```

---

# Follow up
* [LinearRegression](../linear_model/linear_regression.md) - Linear regression model with `.save()`
* [LogisticRegression](../linear_model/logistic_regression.md) - Logistic regression model with `.save()`
* [StandardScaler](../preprocessing/standard_scaler.md) - Feature scaler preserved during save/load
