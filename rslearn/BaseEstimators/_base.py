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

import numpy as np
from numpy import dtype
from rslearn.Errors import *


"""For metrics/* for Multi-output Support"""
def check_multioutput(parameter) -> None:
    valid_params = {
        "weighted",
        "uniform_average",
        "raw_values"
    }

    if parameter not in valid_params:
        raise InvalidValueError(
            f"Got Invalid Parameter, {parameter}. But supported {valid_params} only."
        )

    return


    

"""Numpy Array Convertor from tuple, list, DataFrame"""
def convert_array(arr1, arr2):
    arr1 = np.asarray(arr1, dtype=float)
    arr2 = np.asarray(arr2, dtype=float)

    return arr1, arr2

"""np.array to 1D converter for all rslearn/*"""
def convert1D(*args) -> np.array:
    """
    Parameters
    ----------
    y_true,
    y_pred,
    any, any

    Returns
    -------
    list of Flat Arrays by
    X, y = convert1D(X, y)
    X, y, z = convert1D(X, y, z)

    """
    arrays = []
    for items in args:
        items, _ = convert_array(items, [1012, 1203])
        arrays.append(np.ravel(items))

    return tuple(np.array(arrays))

"""Shape Checker For All rslearn/*"""
def shape_checker(arr1, arr2, output_mode=True):
    arr1, arr2 = convert_array(arr1, arr2)

    # if arr1, arr2 are coming from classification y_true, y_pred
    # to avoid multioutput shape error like
    # Mendontory for Classification Task
    """

    From Regression Output mainly `/metrics/_regression.py`
    y_true = np.array(
        [
            [10, 20],
            [30, 25]
        ]
    )

    y_pred = np.array(
        [
            [10.8, 19.9],
            [30, 23.7]
        ]
    )

    From Classification Output mainly `/linear_model/_Logistic` & `/metrics/_classification.py`


    y_true = np.array(
        [
            [1, 1],
            [0, 1]
        ]
    )

    y_pred = np.array(
        [
            [1, 0],
            [0, 1]
        ]
    )

    """
    if output_mode:
        if arr1.shape != arr2.shape:
            raise InvalidShape(f"Shape mismatch: {arr1.shape} vs {arr2.shape}")

    # if arr1, arr2 are Coming from Regression X, y

    if len(arr1) != len(arr2):  # Assume as X, y
        raise InvalidShape(f"Shape mismatch: {len(arr1)} vs {len(arr2)}")

"""Dimensional Matcher For Only Metrics"""
def dim_validator(arr1):
    """
    Dimensional Matcher For `/metrics/*`
    """
    y_true, _ = convert_array(arr1, [10])  # Converting to Array

    if y_true.ndim == 1:  # Checking direct Dimensional for [1, 2, 3]
        return True

    if y_true.shape[1] == 1:  # Checking Indirect 1D array like [[1], [2], [3]]
        return True

    return False

"""For Logistic Regression Multi-Output Support"""
def multi_output_selector(multi_output_param, scores, weights):
    check_multioutput(multi_output_param)  # Safty Check if Internel Errors Happens

    match multi_output_param:
        case "raw_values":
            return scores

        case "uniform_average":
            return np.mean(scores)

        case "weighted":
            if weights is None:
                raise InvalidValueError("weights must be provided for weighted averaging")

            weights = np.asarray(weights, dtype=float)

            if weights.shape[0] != len(scores):
                raise InvalidValueError("weights length must match number of outputs")

            return np.average(scores, weights=weights)

"""For Linear Regression Weight & Bias Overflow"""
def check_overflow(weights, bias):
    if np.isnan(weights).any() or np.isnan(bias):
        return True

    return False


if __name__ == "__main__":
    X = [[12], [11], [10]]
    y = [[13], [11], [8]]
    X, y = convert1D(X, y)
    print(type(X), y)

    print()

