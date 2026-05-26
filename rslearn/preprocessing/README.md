# Preprocessing

A lightweight and efficient preprocessing module inspired by scikit-learn, 
designed for simplicity, speed, and ease of use.

This module eliminates common friction points while working with feature scaling,
especially for beginners and rapid prototyping.

---

## 🚀 Features

- Minimal and clean API
- Handles both 1D and 2D data automatically
- Works with Python lists and NumPy arrays
- Lightweight alternative to large ML libraries

---

## 📦 Included Components

- `StandardScaler`
- `MinMaxScaler`

> More preprocessing utilities coming soon.

---

## ⚠️ Motivation

While scikit-learn is a powerful and industry-standard library, it can feel
overwhelming for simple tasks due to its size and strict input requirements.

### Common Friction Points

- Requires careful handling of input shapes (1D vs 2D)
- Relatively large import overhead for small tasks

---

## ✅ What This Project Solves

- ✔ Automatic handling of input dimensions (1D / 2D)
- ✔ Accepts both Python lists and NumPy arrays seamlessly
- ✔ Lightweight and fast for small to medium-scale tasks
- ✔ Simple and intuitive structure

---

## 📥 Installation / Usage

```python
from rslearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_data = scaler.fit_transform([1, 2, 3, 4])