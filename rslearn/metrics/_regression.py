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
from rslearn.BaseEstimators._base import (convert_array,
                    check_multioutput, 
                    dim_validator, 
                    shape_checker,
                    multi_output_selector,
                    convert1D)

"""
`_regression.py`, By: Rustam Singh

This File Contains All Important and most used error comupation algorithams,  
like  
- r2_score  
- MAE  
- MSE   
- RMSE   

All Methords
-------------
- `r2_score`    
- `MAE`  
- `MSE`   
- `RMSE`  
"""

# ========================== r2_score =========================
def r2_score(y_true, y_pred, multi_output="uniform_average", weights=None):
    """
    Compute the coefficient of determination (R² score).

    The R² score represents the proportion of variance in the dependent
    variable that is predictable from the independent variables.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.

    multi_output : {"uniform_average", "weighted", "raw_values"}, default="uniform_average"
        Defines how to aggregate scores for multi-output data:
        
        - "uniform_average" : Average scores across all outputs (default)
        - "weighted"        : Weighted average using `weights`
        - "raw_values"      : Return scores for each output separately

    weights : array-like of shape (n_outputs,), optional
        Weights used when `multi_output="weighted"`. Higher values indicate
        greater importance of the corresponding output.

    Returns
    -------
    float or np.ndarray of shape (n_outputs,)
        R² score. Returns a single float for 1D targets or aggregated output,
        or an array of scores when `multi_output="raw_values"`.

    Examples
    --------
    Basic usage with 1D targets:

    >>> from rslearn.metrics import r2_score
    >>> y_true = [3, -0.5, 2, 7]
    >>> y_pred = [2.5, 0.0, 2, 8]
    >>> r2_score(y_true, y_pred)

    Multi-output regression (automatic averaging):

    >>> y_true = [[3, 10], [-0.5, 20], [2, 30], [7, 40]]
    >>> y_pred = [[2.5, 12], [0.0, 18], [2, 33], [8, 39]]
    >>> r2_score(y_true, y_pred)

    Weighted multi-output:

    >>> r2_score(y_true, y_pred, multi_output="weighted", weights=[0.6, 0.4])
    """

    check_multioutput(parameter=multi_output)

    y_true, y_pred = convert_array(y_true, y_pred)

    shape_checker(arr1=y_true, arr2=y_pred, output_mode=True)

    if y_true.size < 2:
        raise ValueError("At least 2 samples are required")

    # Handle 1D case
    if dim_validator(y_true):
        return r2_score_1d(y_true, y_pred)

    # Multi-output case
    r2_scores = r2_score_helper_2D(y_true, y_pred, score_only=True)

    # Handling MultiOutput Stuff

    return multi_output_selector(multi_output_param=multi_output, scores=r2_score, weights=weights)


def r2_score_1d(y_true, y_pred):
    y_true, y_pred = convert1D(y_true, y_pred)

    y_mean = np.mean(y_true)

    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum((y_true - y_mean) ** 2)

    if denominator == 0:
        return 0.0

    return 1 - (numerator / denominator)

def r2_score_helper_2D(y_true, y_pred, weight=None, score_only=False):
    n_col = y_true.shape[1]
    
    r2_scores = []
    for cols in range(n_col):
        selected_true = y_true[:, cols]
        selected_pred = y_pred[:, cols]

        score = r2_score_1d(selected_true, selected_pred)
        r2_scores.append(score)
    
    if score_only:
        return np.array(r2_scores)

    elif weight is None:
        return np.mean(r2_scores)
    
    else:
        if len(weight) != n_col:
            raise ValueError("Invalid weights given")
        
        return np.average(r2_scores, weights=weight)
    
# ========================== r2_score =========================

# ========================== MSE =========================
def mse(
    y_true,
    y_pred,
    multi_output="uniform_average",
    weights=None
):
    
    """
    Mean Squared Error  

    The Mean Squared Error measures the average squared difference between
    the true target values and the predicted values. It penalizes larger
    errors more heavily due to the squaring operation, making it sensitive
    to outliers.

    Parameter
    ---------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Predicted target values.
    
    multi_output : {"uniform_average", "weighted", "raw_values"}, default="uniform_average"
        Defines how to aggregate error for multi-output data:
        
        - "uniform_average" : Average error across all outputs (default)
        - "weighted"        : Weighted average using `weights`
        - "raw_values"      : Return error for each output separately

    weights : array-like of shape (n_outputs,), optional
        Weights used when `multi_output="weighted"`. Higher values indicate
        greater importance of the corresponding output.

    Returns
    -------
    float or np.ndarray of shape (n_outputs,)
        mean squared error Returns a single float for 1D targets or aggregated output
        or an array of scores when `multi_output="raw_values"`.
    
    Raise
    -----
    ValueError: when array's Shape MisMatch or Invalid `multi_output` and Invalid `weights`
    

    Example
    -------
    >>> from rslearn.metrics import mse # Importing
    >>> # 1D case
    >>> print(mse([1, 2, 3], [1, 2, 4]))  # expected: 0.333...

    >>> # multi-output
    >>> y_true = [[1, 2], [3, 4]]
    >>> y_pred = [[1, 3], [2, 5]]

    >>> print(mse(y_true, y_pred, multi_output="raw_values"))
    """
    
    check_multioutput(parameter=multi_output)
    


    # Formula = 1/n sum((y_true - y_pred)**2)

    y_true, y_pred = convert_array(y_true, y_pred)

    shape_checker(arr1=y_true, arr2=y_pred, output_mode=True)
    
    if dim_validator(y_true): # Handling 1D Case
        return mse_helper_1d(y_true, y_pred)
    
    output_errors = mse_helper_2D(y_true, y_pred)

    # Handling Multi-Ouput Sparse Metrics Scene
    return multi_output_selector(multi_output_param=multi_output, scores=output_errors, weights=weights)
    


# MSE helper for 1 columns Simple Regression Ouput
def mse_helper_1d(y_true: np.array, y_pred: np.array):
    y_true, y_pred = convert1D(y_true, y_pred)

    n = len(y_true)

    output_error = np.sum((y_true - y_pred)**2) / n
    return output_error


# MSE helper for multioutput Regression Tasks
def mse_helper_2D(y_true: np.array , y_pred : np.array):
    n_col = y_true.shape[1]
    
    errors = []
    for cols in range(n_col):
        selected_true = y_true[:, cols]
        selected_pred = y_pred[:, cols]

        error = mse_helper_1d(selected_true, selected_pred)
        errors.append(error)
    
    return np.array(errors)
# ========================== MSE =========================


# ========================== RMSE =========================
def rmse(
    y_true,
    y_pred,
    multi_output="uniform_average",
    weights=None
):
    
    """
    Root Mean Squared Error  

    The Root Mean Squared Error measures the average absolute difference between  
    the true target values and the predicted values. It penalizes Smaller Errors Than MSE errors  
    due to the Absolute `np.sqrt` operation, making it not well to outliers as MAE.  

    Parameter
    ---------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Predicted target values.
    
    multi_output : {"uniform_average", "weighted", "raw_values"}, default="uniform_average"
        Defines how to aggregate error for multi-output data:
        
        - "uniform_average" : Average error across all outputs (default)
        - "weighted"        : Weighted average using `weights`
        - "raw_values"      : Return error for each output separately

    weights : array-like of shape (n_outputs,), optional
        Weights used when `multi_output="weighted"`. Higher values indicate
        greater importance of the corresponding output.

    Returns
    -------
    float or np.ndarray of shape (n_outputs,)
        mean squared error Returns a single float for 1D targets or aggregated output
        or an array of scores when `multi_output="raw_values"`.
    
    Raise
    -----
    ValueError: when array's Shape MisMatch or Invalid `multi_output` and Invalid `weights`
    

    Example
    -------
    >>> from rslearn.metrics import rmse # Importing
    >>> # 1D case
    >>> print(rmse([1, 2, 3], [1, 2, 4]))  # expected: 0.578...

    >>> # multi-output
    >>> y_true = [[1, 2], [3, 4]]
    >>> y_pred = [[1, 3], [2, 5]]

    >>> print(rmse(y_true, y_pred, multi_output="raw_values"))
    """
    
    error = mse(y_true, y_pred, multi_output, weights)
    return np.sqrt(error)
    
# ========================== RMSE =========================



# ========================== MAE =========================
def mae(
    y_true,
    y_pred,
    multi_output="uniform_average",
    weights=None
):
    
    """
    Mean Absolute Error  

    The Mean Absolute Error measures the average absolute difference between  
    the true target values and the predicted values. It penalizes Smaller Errors Than MSE errors  
    due to the Absolute `np.abs` operation, making it well to outliers.  

    Parameter
    ---------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Predicted target values.
    
    multi_output : {"uniform_average", "weighted", "raw_values"}, default="uniform_average"
        Defines how to aggregate error for multi-output data:
        
        - "uniform_average" : Average error across all outputs (default)
        - "weighted"        : Weighted average using `weights`
        - "raw_values"      : Return error for each output separately

    weights : array-like of shape (n_outputs,), optional
        Weights used when `multi_output="weighted"`. Higher values indicate
        greater importance of the corresponding output.

    Returns
    -------
    float or np.ndarray of shape (n_outputs,)
        mean squared error Returns a single float for 1D targets or aggregated output
        or an array of scores when `multi_output="raw_values"`.
    
    Raise
    -----
    ValueError: when array's Shape MisMatch or Invalid `multi_output` and Invalid `weights`
    

    Example
    -------
    >>> from rslearn.metrics import mae # Importing
    >>> # 1D case
    >>> print(mae([1, 2, 3], [1, 2, 4]))  # expected: 0.333...

    >>> # multi-output
    >>> y_true = [[1, 2], [3, 4]]
    >>> y_pred = [[1, 3], [2, 5]]

    >>> print(mae(y_true, y_pred, multi_output="raw_values"))
    """
    
    check_multioutput(parameter=multi_output)
    


    # Formula = 1/n sum(np.abs((y_true - y_pred)))

    y_true, y_pred = convert_array(y_true, y_pred)

    shape_checker(arr1=y_true, arr2=y_pred, output_mode=True)
    
    if dim_validator(y_true):
        return mae_helper_1d(y_true, y_pred)
    
    output_errors = mae_helper_2D(y_true, y_pred)

    return multi_output_selector(multi_output_param=multi_output, scores=output_errors, weights=weights)
    
# MAE helper for 1 columns Regression Ouput
def mae_helper_1d(y_true: np.array, y_pred: np.array):
    y_true, y_pred = convert1D(y_true, y_pred)

    n = len(y_true)

    output_error = np.sum(np.abs((y_true - y_pred))) / n

    return output_error


# MAE helper for multioutput Regression Tasks
def mae_helper_2D(y_true: np.array , y_pred : np.array):
    n_col = y_true.shape[1]
    
    errors = []
    for cols in range(n_col):
        selected_true = y_true[:, cols]
        selected_pred = y_pred[:, cols]

        error = mae_helper_1d(selected_true, selected_pred)
        errors.append(error)
    
    return np.array(errors)

# ========================== MAE =========================
