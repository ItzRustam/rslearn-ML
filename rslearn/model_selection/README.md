# 📊 Model Selection Module

This module contains utilities for splitting datasets for training and evaluation.

---

## 🔹 train_test_split

The `train_test_split` function is used to split datasets into training and testing subsets. It supports both standard random splitting and stratified splitting for imbalanced datasets.

---

## ⚡ Features

* ✅ Random data splitting
* ✅ Stratified splitting (maintains class distribution)
* ✅ Reproducibility using `random_state`
* ✅ Supports multiple input types:

  * NumPy arrays
  * Python lists / tuples
  * pandas DataFrame / Series

---

## 🧠 Why use Stratified Splitting?

In imbalanced datasets, normal splitting can lead to uneven class distribution.

Example:

```python id="m0x8lx"
y = [0, 0, 0, 0, 1, 1]
```

A normal split may result in:

* Train: only class `0`
* Test: only class `1` ❌

Using `stratify=y` ensures:

* Same class distribution in both train and test sets ✅

---

## 🧪 Usage

```python id="1w2fxl"
from rslearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

---

## 📥 Parameters

| Parameter      | Type       | Description                                 |
| -------------- | ---------- | ------------------------------------------- |
| `X`            | array-like | Feature data                                |
| `y`            | array-like | Target labels                               |
| `test_size`    | float      | Proportion of test data (0 < test_size < 1) |
| `random_state` | int        | Controls randomness                         |
| `stratify`     | array-like | Ensures balanced class distribution         |

---

## 📤 Returns

```python id="r9pnqk"
X_train, X_test, y_train, y_test
```

All outputs are NumPy arrays.

---

## ⚠️ Notes

* `X` and `y` must have the same number of samples
* `test_size` must be between 0 and 1
* Each class in `stratify` must have at least 2 samples
* Internally, all inputs are converted to NumPy arrays

---

## 🚀 Future Additions

* KFold
* StratifiedKFold
* ShuffleSplit

---

## 👨‍💻 Author

Rustam Singh Bhadouriya
