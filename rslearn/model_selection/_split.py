# rslearn-ML
# Copyright (C) 2026 Rustam Singh Bhadouriya
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#
# See the LICENSE file for more details.

"""
_split.py, By: Rustam Singh Bhadouriya

This module contains the function `train_test_split` for splitting data
into training and testing sets to help with model training and reduce overfitting.

--------------------------------------------

Features
--------
- Stratified splitting support (like sklearn)
- Standard random splitting
- Supports multiple data types (list, tuple, pandas DataFrame, NumPy array)

"""

import numpy as np


def train_test_split(X, y, test_size: float = 0.2, random_state: int = None, stratify=None):
    """
    Split arrays into random train and test subsets.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature input data. Can be a NumPy array, list, tuple, or pandas DataFrame.

    y : array-like of shape (n_samples,)
        Target labels corresponding to X. Must have the same length as X.

    test_size : float, default=0.2
        Proportion of the dataset to include in the test split.
        Must be between 0 and 1.

    random_state : int, default=None
        Controls the randomness for reproducibility.

    stratify : array-like, default=None
        If not None, data is split in a stratified fashion using this array.
        Typically, this should be the target variable (y).

    Returns
    -------
    X_train, X_test, y_train, y_test : ndarray
        Split data arrays.

    Example
    -------
    >>> from rslearn.model_selection import train_test_split
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     X, y, test_size=0.3, random_state=42, stratify=y
    ... )
    """

    # Validate test_size
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")

    # Convert inputs to NumPy arrays
    X = np.asarray(X)
    y = np.asarray(y)

    # Validate input sizes
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")

    # Set random seed
    if random_state is not None:
        np.random.seed(random_state)

    # Stratified split
    if stratify is not None:
        if len(X) != len(stratify):
            raise ValueError(f"Inputs contain different shapes: {(len(X), len(stratify))}")

        stratify = np.asarray(stratify)
        unique_classes, counts = np.unique(stratify, return_counts=True)

        # Ensure each class has at least 2 samples
        if np.any(counts < 2):
            raise ValueError("Each class in stratify must have at least 2 samples")

        train_indices = []
        test_indices = []

        for cls in unique_classes:
            cls_indices = np.where(stratify == cls)[0]
            shuffled = np.random.permutation(cls_indices)

            cls_size = len(shuffled)
            split_size = max(1, int(test_size * cls_size))
            split_size = min(split_size, cls_size - 1)

            train_part = shuffled[split_size:]
            test_part = shuffled[:split_size]

            train_indices.extend(train_part)
            test_indices.extend(test_part)

        train_indices = np.array(train_indices, dtype=np.int64)
        test_indices = np.array(test_indices, dtype=np.int64)

        # Final shuffle
        train_indices = np.random.permutation(train_indices)
        test_indices = np.random.permutation(test_indices)

        return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

    # Standard random split (no stratification)
    data_size = X.shape[0]
    shuffled_indices = np.random.permutation(data_size)
    split_size = int(test_size * data_size)

    train_indices = shuffled_indices[split_size:]
    test_indices = shuffled_indices[:split_size]

    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


if __name__ == "__main__":
    print("train_test_split module loaded successfully.")