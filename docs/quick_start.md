# Quick Start Guide: rslearn-ML

> Fast-track practical guide to getting started with `rslearn-ML`.

This guide provides runnable code examples for the core capabilities of `rslearn-ML`. For complete parameter documentation and mathematical formulations, explore the <a href="user_guide/README.md">User Guide</a>.

---

## Quick Reference Summary

| Task | Primary Class / Function | Module | Example Action |
|---|---|---|---|
| **Linear Regression** | `LinearRegression` | `rslearn.linear_model` | `model.fit(X_train, y_train)` |
| **Logistic Classification** | `LogisticRegression` | `rslearn.linear_model` | `model.fit(X_train, y_train)` |
| **K-Nearest Neighbors** | `KNNClassifier`, `KNNRegressor` | `rslearn.neighbors` | `model.predict(X_test)` |
| **Automated Pipeline** | `pipeline` | `rslearn.Pipeline` | `pipe.fit(X, y)` |
| **Feature Standardization** | `StandardScaler`, `MinMaxScaler` | `rslearn.preprocessing` | `scaler.fit_transform(X)` |
| **Dataset Partitioning** | `train_test_split` | `rslearn.model_selection` | `X_tr, X_te, y_tr, y_te = train_test_split(X, y)` |
| **Universal Evaluation** | `evaluate_model` | `rslearn.metrics` | `evaluate_model(model=m, X=X_te, y_true=y_te)` |
| **Model Persistence** | `load_model` | `rslearn.loader` | `m = load_model("model.rslr")` |

---

## 1. Installation & Environment Setup

Install or upgrade `rslearn-py` from PyPI:

```bash
pip install --upgrade rslearn-py
```

Verify your installation and version:

```python
import rslearn

print("PyPI Version:  ", rslearn.__version__)
print("GitHub Version:", rslearn.__github_version__)
```

---

## 2. Linear Regression (Continuous Target)

Train a gradient-descent linear regression model with automatic feature scaling, evaluate metrics in one line, and persist the model to disk:

```python
import numpy as np
from rslearn.linear_model import LinearRegression
from rslearn.model_selection import train_test_split

# 1. Generate synthetic housing feature data
np.random.seed(42)
X = np.random.uniform(500, 3500, size=(150, 3))  # sqft, bedrooms, age
y = 150.0 * X[:, 0] + 20000.0 * X[:, 1] - 1200.0 * X[:, 2] + 50000.0 + np.random.normal(0, 5000, size=150)

# 2. Partition into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Initialize and fit model (auto-scaling is enabled by default)
model = LinearRegression(regulization="l2", alpha=0.1, lr=0.01, max_itr=3000)
model.fit(X_train, y_train, scale=True)

# 4. Predict and evaluate
predictions = model.predict(X_test)
evals = model.evaluate(y_pred=predictions, y_true=y_test)

print("Linear Regression Results:")
print(f"R² Score: {evals['evaluation']['r2_score']:.4f}")
print(f"MSE:      {evals['evaluation']['mse']:.4f}")
print(f"MAE:      {evals['evaluation']['mae']:.4f}")
print(f"RMSE:     {evals['evaluation']['rmse']:.4f}")

# 5. Save model to disk
model.save("housing_model.rsl")
print("Saved model to housing_model.rslr")
```

---

## 3. Logistic Regression (Classification)

Train a binary or multi-class classifier using automatic solver selection:

```python
import numpy as np
from rslearn.linear_model import LogisticRegression
from rslearn.model_selection import train_test_split

# 1. Prepare synthetic classification data
np.random.seed(42)
X = np.random.randn(200, 2)
y = (1.5 * X[:, 0] - 2.0 * X[:, 1] > 0).astype(int)

# 2. Split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 3. Train classifier
clf = LogisticRegression(solver="auto", lr=0.05, max_itr=3000)
clf.fit(X_train, y_train, scale=True)

# 4. Predict class labels and probabilities
preds = clf.predict(X_test)
probs = clf.predict_proba(X_test)

# 5. Evaluate all classification metrics
results = clf.evaluate(y_pred=preds, y_true=y_test)
print("Classification Results:")
print(f"Accuracy:  {results['evaluation']['accuracy_score']:.4f}")
print(f"Precision: {results['evaluation']['precision']:.4f}")
print(f"Recall:    {results['evaluation']['recall']:.4f}")
print(f"F1 Score:  {results['evaluation']['f1_score']:.4f}")

# 6. Save classifier
clf.save("classifier.rsl")
print("Saved classifier to classifier.rslc")
```

---

## 4. K-Nearest Neighbors (Instance Learning)

Use `KNNClassifier` for voting-based classification:

```python
import numpy as np
from rslearn.neighbors import KNNClassifier
from rslearn.model_selection import train_test_split

# Multi-class data
X = np.array([
    [1.0, 1.1], [1.2, 0.9], [0.8, 1.0],
    [5.0, 5.2], [5.1, 4.9], [4.8, 5.0],
    [9.0, 8.8], [9.2, 9.1], [8.9, 9.0]
])
y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])

# Initialize KNN Classifier (k=3)
knn = KNNClassifier(k_neighbors=3, hard_scale_off=False)
knn.fit(X, y, scale=True)

# Predict new points
new_points = np.array([[1.1, 1.0], [5.2, 5.0], [9.1, 9.0]])
knn_preds = knn.predict(new_points)
print("KNN Predictions:", knn_preds)  # Output: [0, 1, 2]
```

---

## 5. End-to-End Automated Pipeline

Automate feature scaling, train-test splitting, training, and evaluation in one object:

```python
import numpy as np
from rslearn.Pipeline import pipeline
from rslearn.linear_model import LinearRegression
from rslearn.preprocessing import StandardScaler

# Raw feature dataset
X = np.array([
    [10.0, 20.0], [40.0, 50.0], [15.0, 15.0],
    [40.0, 35.0], [12.0, 15.0], [25.0, 15.0],
    [13.0, 12.0], [15.0, 10.0], [15.0, 15.0]
])
y = np.array([30.0, 90.0, 30.0, 75.0, 27.0, 40.0, 25.0, 25.0, 30.0])

# Configure pipeline with automated validation split
pipe = pipeline(
    params={
        "model": LinearRegression(regulization="l1"),
        "scaler": StandardScaler()
    },
    validation_split=True,
    split_params={"test_size": 0.25, "random_state": 42, "stratify": None}
)

# Fit pipeline: scales features, fits estimator, and displays validation report
eval_output, X_val = pipe.fit(X, y)

# Predict on new unscaled raw inputs
raw_new_data = np.array([[20.0, 25.0]])
pipe_prediction = pipe.predict(raw_new_data)
print("Pipeline Prediction on Raw Data:", pipe_prediction)
```

---

## 6. Restoring Saved Models

Restore pre-trained models from `.rslr` or `.rslc` files and predict immediately without retraining:

```python
from rslearn.loader import load_model
import numpy as np

# Load linear regression model
loaded_reg = load_model("housing_model.rslr")

# Predict on new features (historical scaler parameters are applied automatically)
new_houses = np.array([[2200.0, 3.0, 10.0], [1600.0, 2.0, 5.0]])
estimates = loaded_reg.predict(new_houses)
print("Loaded Model Housing Estimates:", estimates)
```

---

## Next Steps

To dive deeper into any module, consult the specific chapters:
* <a href="user_guide/README.md">Complete User Guide</a>
* <a href="user_guide/linear_model/README.md">Linear Models & Regularization</a>
* <a href="user_guide/neighbors/README.md">K-Nearest Neighbors Family</a>
* <a href="user_guide/pipeline/README.md">End-to-End Pipelines</a>
* <a href="user_guide/preprocessing/README.md">Feature Scalers</a>
* <a href="user_guide/metrics/README.md">Evaluation Metrics</a>
* <a href="user_guide/loader/README.md">Model Persistence</a>
* <a href="contributor_guide/README.md">Contributor Guide</a>
