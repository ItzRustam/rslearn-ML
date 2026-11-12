# linear_model.LogisticRegression

## Overview
`LogisticRegression` is a classification algorithm optimized using Gradient Descent. It models the probability of class membership for both binary and multi-class categorical problems using the sigmoid function and One-vs-Rest (OvR) multi-model ensembles.

**Features**  
* Supports both binary classification (`solver="liblinear"`) and multi-class classification (`solver="saga"`).
* Automatic solver selection (`solver="auto"`) that determines the optimal solver based on target label cardinality.
* Integrated feature scaling via `StandardScaler` (`scale=True`) or `backupScaler` (`scale=False`).
* Full user control over feature scaling using `hard_scale_off=True`.
* Native probability estimation via `.predict_proba(X)`.
* Comprehensive all-in-one evaluation via `.evaluate()` returning accuracy, precision, recall, and F1 score.
* Model serialization and persistence in compressed `.rslc` format.

---

## Parameters

### Class Parameters (tuning parameters)

| Parameter | Type | Default | Description |
|---|---|---|---|
| `solver` | `str` | `"auto"` | Optimization solver strategy: `"liblinear"` for binary classification, `"saga"` for multi-class categorical classification, or `"auto"` for automatic detection based on `y`. |
| `lr` | `float` | `0.001` | Learning rate (step size) for gradient descent weight updates. |
| `max_itr` | `int` | `3000` | Maximum number of iterations for the gradient descent optimization loop. |
| `weights` | `np.ndarray` or `None` | `None` | Initial weight vector. If `None`, initialized randomly from a uniform distribution $[0.2, 3.0]$. |
| `bias` | `float` or `None` | `None` | Initial bias scalar. If `None`, initialized to `0.0`. |
| `hard_scale_off` | `bool` | `False` | When `True`, completely disables internal feature scaling routines. |
| `catogirical_model` | `list` of `LogisticRegression` | `[]` | Internal list of binary submodels used by the `"saga"` multi-class solver. Typically populated internally during `fit()` or `load_model()`. |

**Usage**  
```python
from rslearn.linear_model import LogisticRegression

# Binary or multi-class with automatic solver detection
model = LogisticRegression(
    solver="auto",
    lr=0.01,
    max_itr=4000,
    hard_scale_off=False
)
```

---

## NOTE
1. **Feature Scaling**: Gradient descent step sizes are directly proportional to feature scales. Leaving `hard_scale_off=False` and `scale=True` ensures internal standardization with `StandardScaler`, yielding faster convergence and higher accuracy.
2. **Automatic Solver Selection**: When `solver="auto"` (default), `LogisticRegression` counts unique classes in target `y`. If exactly 2 classes exist, it automatically switches to `"liblinear"`. If 3 or more classes exist, it switches to `"saga"`.
3. **Class Imbalance Safeguard**: In `"liblinear"` binary mode, the gradient calculation automatically balances class loss weights based on the proportions of positive and negative classes in `y`.
4. **Serialization Extension**: Calling `model.save("model.rsl")` saves the model in `.rslc` format (the `'c'` denotes classification).

---

### Method Parameters

| Method | Parameters | Return Type | Description |
|---|---|---|---|
| `fit(X, y, scale=True)` | `X`: Feature matrix (`n_samples, n_features`)<br>`y`: Target labels (`n_samples,`)<br>`scale`: `bool`, default `True` | `None` | Trains the logistic regression classifier using gradient descent. |
| `predict(X)` | `X`: Query feature matrix (`n_samples, n_features`) | `np.ndarray` | Predicts discrete class labels for input observations. |
| `predict_proba(X)` | `X`: Query feature matrix (`n_samples, n_features`) | `np.ndarray` | Computes class probability distributions of shape `(n_samples, n_classes)`. |
| `evaluate(X=None, y_pred=None, y_true=None)` | `X`: Optional feature matrix<br>`y_pred`: Optional precomputed predictions<br>`y_true`: Ground truth target labels (`n_samples,`) | `dict` | Computes classification metrics (`accuracy_score`, `precision`, `recall`, `f1_score`). Returns `{"prediction": ..., "evaluation": ...}`. |
| `get_weight_bias()` | None | `tuple (np.ndarray, float)` | Returns the learned `(weights, bias)` for binary models. |
| `save(file_name)` | `file_name`: `str`, default `"rslearn_model.rsl"` | `str` | Serializes model parameters, solver options, and scaler statistics into `.rslc` format. |

---

## Mathematical Formulation

### 1. Hypothesis & Sigmoid Function
For binary classification ($y \in \{0, 1\}$), the linear combination of inputs is mapped to a probability via the logistic sigmoid function:
$$z = X \cdot w + b$$
$$\hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}} = P(y = 1 \mid X)$$

### 2. Weighted Gradient Descent
The loss is minimized using balanced gradients to account for class representation:
$$\text{weight}_{\text{pos}} = \frac{N}{2 \sum y}, \quad \text{weight}_{\text{neg}} = \frac{N}{2 \sum (1 - y)}$$
$$w_{\text{factor}} = y \cdot \text{weight}_{\text{pos}} + (1 - y) \cdot \text{weight}_{\text{neg}}$$
$$\frac{\partial \mathcal{L}}{\partial w} = \frac{1}{N} X^T \Big( w_{\text{factor}} \odot (\hat{y} - y) \Big)$$
$$\frac{\partial \mathcal{L}}{\partial b} = \frac{1}{N} \sum \Big( w_{\text{factor}} \odot (\hat{y} - y) \Big)$$

Parameters are updated as:
$$w \leftarrow w - \eta \cdot \frac{\partial \mathcal{L}}{\partial w}, \quad b \leftarrow b - \eta \cdot \frac{\partial \mathcal{L}}{\partial b}$$

### 3. Multi-Class One-vs-Rest (OvR)
For multi-class classification with $C$ classes ($y \in \{0, 1, \dots, C-1\}$):
* The `"saga"` solver trains $C$ distinct binary `liblinear` classifiers.
* Classifier $c$ predicts whether sample $x$ belongs to class $c$ ($y = 1$) or not ($y = 0$).
* Probabilities across all $C$ models are concatenated and normalized:
  $$P(y = c \mid X) = \frac{P_c(X)}{\sum_{j=1}^C P_j(X)}$$
* The final class prediction selects the class with the maximum posterior probability:
  $$\hat{y} = \arg\max_c P(y = c \mid X)$$

---

## Usage

### 1. Binary Classification Example (`liblinear` Solver)

```python
import numpy as np
from rslearn.linear_model import LogisticRegression
from rslearn.model_selection import train_test_split

# 1. Create synthetic binary classification dataset
np.random.seed(42)
n_samples = 200
X = np.random.randn(n_samples, 2)
# Decision boundary: 1.5 * x0 - 2.0 * x1 + 0.3 > 0
logits = 1.5 * X[:, 0] - 2.0 * X[:, 1] + 0.3
y = (logits > 0).astype(int)

# 2. Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 3. Train binary LogisticRegression model
clf = LogisticRegression(solver="liblinear", lr=0.05, max_itr=3000)
clf.fit(X_train, y_train, scale=True)

# 4. Generate discrete class predictions and probability estimates
predictions = clf.predict(X_test)
probabilities = clf.predict_proba(X_test)

print("First 5 Discrete Predictions:", predictions[:5])
print("First 5 Probabilities [P(y=0), P(y=1)]:\n", probabilities[:5])

# 5. Evaluate classification performance in one call
eval_result = clf.evaluate(X=X_test, y_true=y_test)
print("\nEvaluation Summary:")
print(f"Accuracy:  {eval_result['evaluation']['accuracy_score']:.4f}")
print(f"Precision: {eval_result['evaluation']['precision']:.4f}")
print(f"Recall:    {eval_result['evaluation']['recall']:.4f}")
print(f"F1 Score:  {eval_result['evaluation']['f1_score']:.4f}")

# 6. Save model to disk
clf.save("binary_classifier.rsl")
print("Model saved as binary_classifier.rslc")
```

---

### 2. Multi-Class Classification Example (`saga` Solver)

```python
import numpy as np
from rslearn.linear_model import LogisticRegression
from rslearn.model_selection import train_test_split
from rslearn.loader import load_model

# 1. Create synthetic 3-class dataset (Classes 0, 1, 2)
np.random.seed(42)
cluster_0 = np.random.randn(50, 2) + np.array([-3.0, -3.0])
cluster_1 = np.random.randn(50, 2) + np.array([0.0, 3.0])
cluster_2 = np.random.randn(50, 2) + np.array([3.0, -3.0])

X = np.vstack([cluster_0, cluster_1, cluster_2])
y = np.array([0] * 50 + [1] * 50 + [2] * 50)

# 2. Split dataset with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 3. Initialize with solver="auto" (will auto-select "saga" for 3 classes)
multi_clf = LogisticRegression(solver="auto", lr=0.05, max_itr=4000)
multi_clf.fit(X_train, y_train, scale=True)

# 4. Predict and evaluate
test_preds = multi_clf.predict(X_test)
results = multi_clf.evaluate(y_pred=test_preds, y_true=y_test)
print(f"Multi-Class Accuracy: {results['evaluation']['accuracy_score']:.4f}")

# 5. Save and reload multi-class model
multi_clf.save("multiclass_model.rsl")

loaded_clf = load_model("multiclass_model.rslc")
restored_preds = loaded_clf.predict(X_test)
print("Restored model predictions match:", np.array_equal(test_preds, restored_preds))
```

---

# Follow up
* [LinearRegression](linear_regression.md) - Continuous regression modeling
* [Regularizations (Lasso, Ridge, ElasticNet)](regulizations.md) - Overfitting prevention techniques
* [Classification Metrics](../metrics/classification.md) - Accuracy, confusion matrix, precision, recall, and F1 score
* [StandardScaler](../preprocessing/standard_scaler.md) - Feature standardization for gradient descent
* [train_test_split](../model_selection/splitting.md) - Dataset partitioning with stratification
* [load_model](../loader/load_model.md) - Restoring pre-trained `.rslc` classifiers
