# 🚀 rslearn

> A beginner-friendly machine learning library that automates preprocessing, training, and evaluation.

![License](https://img.shields.io/badge/license-GPLv3-blue.svg)
![Python](https://img.shields.io/badge/python-3.13.x-blue.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)
![Contributions](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)

---

## ✨ Why rslearn?

- ⚡ Minimal setup — no complex configuration  
- 🤖 Automatic pipeline (scaling, splitting, evaluation)  
- 📊 Built-in metrics for regression & classification  
- 🧠 Designed for beginners learning ML concepts  
- 🧩 Clean and simple API inspired by sklearn  

---

## Release & Changes
* **Version : 1.0.7 - 1.0.1**
* **Release Date: 2026-05-26**

## 🚀 Features

### Latest (In Pipeline & linear_model): 
* `Pipeline With Inbuilt Analysis Method With Regulizations class support`
* `evaluation() Function Support in All Classes`

More Info: [CHANGELOG](CHANGELOG.md)  
More Parameter Info (in Pipeline): [README](rslearn/Pipeline/README.md)  
More Parameter Info (in linear_models): [README](rslearn/linear_model/README.md)
Read Doc Strings For Extra Information About Parameter

### Fix
* Shape varification issue in `linear_model`
* Auto Scaler Problem

### Changed
* MIT License to GNU GPL v3
* `analysis()` to `evaluate()` in Pipeline

## 🗄️ New File & Folders
* **Folder: Pipeline**
* **File: Pipeline/_pipeline.py**

## Download Version Specific Module
***[Downloads - Module](download.md)***

### 📊 Linear Models

* Linear Regression (Single & Multi-feature)
* Logistic Regression (Binary & Multi-class)
* Ridge Regression (L2 Regularization)
* Lasso Regression (L1 Regularization)
* Elastic Net (L1 + L2)

---

### 📏 Metrics

* Mean Squared Error (MSE)
* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* R² Score
* Accuracy (for classification)

✔ Supports **single-output and multi-output** tasks

---

### 🔧 Preprocessing

* StandardScaler
* MinMaxScaler

---

### 🧪 Model Selection

* Train-Test Split

  * Supports `stratify` for balanced sampling

---

## ⚙️ Optimization Details

All models in **rslearn** are implemented using **Gradient Descent**.

⚠️ **Important:**

* Feature scaling is highly recommended for stable and faster convergence.
* Use:

  * `StandardScaler` (recommended)
  * or `MinMaxScaler`

---



## 🤖 Auto Standard Scaling (Linear, Logistic, Ridge, Lasso, ElasticNet)

models include Inbuilt StandardScaler Feature in fit() Method:

```python
scale=True  # default
```

* Automatically applies feature scaling internally
* Helps prevent numerical instability

---

## 📁 Project Structure

```
rslearn/
│
├── linear_model/
│   ├── _linear_regression.py
│   ├── _logistic_regression.py
│   ├── _ridge.py
│   ├── _lasso.py
│   ├── _elastic_net.py
│
├── preprocessing/
│   ├── _scaler.py
│
├── metrics/
│   ├── _regression.py
│
├── model_selection/
│   ├── _split.py
│
└── README.md
```

📌 Each module contains its own **detailed README** with usage examples and explanations.

---

## 🛠️ Installation

### Clone the repository

```bash
git clone https://github.com/ItzRustam/rslearn-ML.git
cd rslearn-ML/
```

### Install Usable Library (Stable - Latest)
``` bash
pip install rslearn-py
```
## Download Version Specific Module
***[Downloads Older Library](download.md)***

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## 📌 Quick Example

```python
from rslearn.linear_model import LinearRegression
from rslearn.preprocessing import StandardScaler
import numpy as np

X = np.array([10, 20, 30])
y = np.array([5, 10, 15])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LinearRegression()
model.fit(X_scaled, y)

print(model.predict([40]))
```

---

## 📚 Documentation

* Each folder includes its own **README.md**
* Covers:

  * Usage
  * Parameters
  * Examples
  * Internal working

---

## 🎯 Goals of this Project

* Understand ML algorithms from scratch
* Build a sklearn-like API
* Create reusable and modular ML components
* Learn real-world ML system design
* Check Self Ability

---

## 🧑‍💻 Author

**ItzRustam**

---

## 📜 License

This project is licensed under the GNU GPL v3 License.
