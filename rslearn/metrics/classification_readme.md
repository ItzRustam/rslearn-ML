# 📊 rslearn.metrics

This module provides essential Classification metrics implemented in NumPy.

Currently, it includes:

- **Accuracy Score (`accuracy_score`)**
- **Confusion Metrics (`confusion_metrics`)**
- **Recall (`recall`)**
- **Precision (`precision`)** 
- **F1 Score (`f1_score`)**

Upcoming: 
- **Classification Report (`classification_report`)**


# `Features`
* Supports Binary and Multi output both
* scale between 0-1 like sklearn
* Multi Data type supported (`list`, `tuple`, `np.array`)
* Auto shape convertion E.g `(n, 1) -> (n, )`
* Auto Detect Multi Output


# `Comman Parameters`

* `y_true` : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) labels. Multi class also supported.    

* `y_pred` : 1d array-like, or label indicator array / sparse matrix
        Predicted labels, as returned by a classifier.

# `Note`
* Make Sure Both y_true, y_pred length are same

# 🔹 `accuracy_score`

## Parameters
    
* `y_true` : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) labels. Sparse matrix is only supported when
        labels are of :term:`multilabel` type.

* `y_pred` : 1d array-like, or label indicator array / sparse matrix
        Predicted labels, as returned by a classifier. Sparse matrix is only
        supported when labels are of :term:`multilabel` type.

* `normalize` : bool, default=True
        If ``False``, return the number of correctly classified samples.
        Otherwise, return the fraction of correctly classified samples.

* `weights` : array-like of shape (n_samples,), default=None
        Sample weights.
    
* `multi_output` : {"uniform_average", "weighted", "raw_values"}, default="uniform_average"
        Defines how to aggregate scores for multi-output data:
        
    - "uniform_average" : Average scores across all outputs (default)
    - "weighted"        : Weighted average using `weights`
    - "raw_values"      : Return scores for each output separately

## Code

``` python

from rslearn.metrics import accuracy_score

y_true = [[3, 10], [-0.5, 20], [2, 30], [7, 40]]
y_pred = [[2.5, 12], [0.0, 18], [2, 33], [8, 39]]

print(f"Score: {accuracy_score(y_true, y_pred)}")

```

We Took Example of Multi output feel free to use anything for it!


# 🔹 `Confusion Metrics`

Uses Common Parameter y_true, y_pred

`Returns`: **n_class X n_class metrics**

## Code

``` python
from rslearn.metrics import confusion_metrics

cm = confusion_metrics(y_true, y_pred)
print(cm)
```

- Supports Multi-output & Binary-output both


## 🔹 `Recall`, `precision`, `f1_score`

All of These Works on Confusion Metrics

- All Works On Same Parameter y_true, y_pred explain Above at #Common Parameters

``` python
from rslearn.metrics import recall, precision, f1_score

recall_score = recall(y_true, y_pred)
precision_score = precision(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

# All Of these returns float64

print(f"Recall: {recall_score}")
print(f"Precision: {precision}")
print(f"F1 Score: {f1}")
```

### `More Coming Soon`