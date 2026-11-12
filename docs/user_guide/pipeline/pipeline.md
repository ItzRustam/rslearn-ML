# Pipeline.pipeline

## Overview
`pipeline` chains preprocessing, estimator training, automatic validation partitioning, and statistical evaluation into a single cohesive interface. It eliminates the boilerplate of tracking multiple transformations across train and test sets.

**Features**  
* Automates feature scaling with `StandardScaler` or `MinMaxScaler`.
* Automatic train-test splitting and immediate validation testing with `validation_split=True`.
* Built-in multi-metric reporting for both regression and classification models.
* Raw unscaled data inference: `.predict(new_data)` scales input internally before feeding it to the estimator.
* Supports all `rslearn` estimators (`LinearRegression`, `LogisticRegression`, `Ridge`, `Lasso`, `ElasticNet`, `KNNClassifier`, `KNNRegressor`).

---

## Parameters

### Class Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `params` | `dict` | `{}` | Dictionary specifying the estimator (`"model"`) and optional scaler (`"scaler"`). *Required*. |
| `validation_split` | `bool` | `False` | When `True`, automatically partitions data, fits on train split, predicts on test split, and reports evaluation metrics. |
| `split_params` | `dict` | `{"test_size": 0.25, "random_state": 67, "stratify": None}` | Splitting configuration applied when `validation_split=True`. |

### `params` Dictionary Specification

| Key | Expected Type | Description | Example |
|---|---|---|---|
| `"model"` | Estimator instance | Model to train (*Required*) | `LinearRegression()`, `LogisticRegression()`, `KNNClassifier()` |
| `"scaler"` | Scaler instance | Preprocessor for scaling features (*Optional*) | `StandardScaler()`, `MinMaxScaler()` |

### `split_params` Configuration

| Key | Type | Default | Description |
|---|---|---|---|
| `"test_size"` | `float` | `0.25` | Proportion of data reserved for validation. |
| `"random_state"` | `int` | `67` | Seed controlling random split shuffle. |
| `"stratify"` | `array-like` | `None` | Labels for stratified partitioning in classification. |

**Usage**  
```python
from rslearn.Pipeline import pipeline
from rslearn.linear_model import LinearRegression
from rslearn.preprocessing import StandardScaler

pipe = pipeline(
    params={
        "model": LinearRegression(regulization="l1"),
        "scaler": StandardScaler()
    },
    validation_split=True,
    split_params={"test_size": 0.2, "random_state": 42, "stratify": None}
)
```

---

## NOTE
1. **Model Scaling Disabled Inside Pipeline**: When a scaler is provided in `params`, `pipeline` automatically manages feature scaling and sets the model's `hard_scale_off=True` to avoid redundant double-scaling.
2. **Auto-inverse Scaling Output**: When `validation_split=True`, `fit()` returns both the evaluation results dictionary and the unscaled `X_test` features restored via `inverse_transform`.

---

### Method Parameters

| Method | Parameters | Return Type | Description |
|---|---|---|---|
| `fit(X, y, verbose=True)` | `X`: Feature matrix<br>`y`: Target array<br>`verbose`: `bool`, default `True` | `(output, X_test)` if `validation_split=True`, else `None` | Scales data (if scaler configured), trains model on train split, and optionally evaluates on test split. |
| `predict(new_data)` | `new_data`: Feature matrix matching `X` shape | `np.ndarray` | Scales `new_data` using fitted scaler and returns predictions from the trained model. |
| `evaluate(X=None, y_pred=None, y_true=None)` | `X`: Feature matrix<br>`y_pred`: Precomputed predictions<br>`y_true`: Target values | `dict` | Computes full suite of task metrics on the trained pipeline. |

---

## Usage

### 1. Workflow with `validation_split=True` (Automatic Split, Train & Evaluate)

```python
import numpy as np
from rslearn.Pipeline import pipeline
from rslearn.linear_model import LinearRegression
from rslearn.preprocessing import StandardScaler

# Sample data
X = np.array([
    [10.0, 20.0],
    [40.0, 50.0],
    [15.0, 15.0],
    [40.0, 35.0],
    [12.0, 15.0],
    [25.0, 15.0],
    [13.0, 12.0],
    [15.0, 10.0],
    [15.0, 15.0]
])
y = np.array([30.0, 90.0, 30.0, 75.0, 27.0, 40.0, 25.0, 25.0, 30.0])

# Configure pipeline with auto-scaling and automatic validation splitting
pipe = pipeline(
    params={
        "model": LinearRegression(regulization="l2", alpha=0.1),
        "scaler": StandardScaler()
    },
    validation_split=True,
    split_params={"test_size": 0.25, "random_state": 42, "stratify": None}
)

# Fitting automatically prints predictions, evaluations, and returns output
results, test_features = pipe.fit(X=X, y=y, verbose=True)

# Generate predictions on brand-new unseen raw observations
new_samples = np.array([[20.0, 25.0], [35.0, 30.0]])
new_predictions = pipe.predict(new_samples)
print("New Predictions:", new_predictions)
```

### 2. Workflow with `validation_split=False` (Fit on Full Dataset)

```python
import numpy as np
from rslearn.Pipeline import pipeline
from rslearn.linear_model import LogisticRegression
from rslearn.preprocessing import MinMaxScaler

# Classification dataset
X = np.array([[1.0, 2.0], [1.5, 1.8], [5.0, 8.0], [6.0, 9.0]])
y = np.array([0, 0, 1, 1])

pipe = pipeline(
    params={
        "model": LogisticRegression(solver="liblinear"),
        "scaler": MinMaxScaler()
    },
    validation_split=False
)

# Fit on all provided data
pipe.fit(X=X, y=y)

# Predict directly
preds = pipe.predict([[1.2, 1.9], [5.5, 8.5]])
print("Classification predictions:", preds)
```

---

# Follow up
* [StandardScaler](../preprocessing/standard_scaler.md) - Feature standardization scaler
* [MinMaxScaler](../preprocessing/minmax_scaler.md) - Range-bounded feature scaler
* [LinearRegression](../linear_model/linear_regression.md) - Linear regression estimator
* [LogisticRegression](../linear_model/logistic_regression.md) - Classification estimator
