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
_classification.py  

This File Contains Many Important tools and sets of algorithams for classification tasks like  

- `Accuracy Score`
- `log loss`
- `classification report`
- `precesion score`
- `recall score`

"""

import numpy as np
from rslearn.BaseEstimators._base import (convert_array,
                    check_multioutput,
                    dim_validator,
                    shape_checker,
                    multi_output_selector,
                    convert1D)

def accuracy_score(y_true, y_pred, weights=None, multi_output="uniform_average", normalize=True):
    """Accuracy classification score.

    In multilabel classification, this function computes subset accuracy:
    the set of labels predicted for a sample must *exactly* match the
    corresponding set of labels in y_true.

    Read more in the `Documentation or README.md`.

    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) labels. Sparse matrix is only supported when
        labels are of :term:`multilabel` type.

    y_pred : 1d array-like, or label indicator array / sparse matrix
        Predicted labels, as returned by a classifier. Sparse matrix is only
        supported when labels are of :term:`multilabel` type.

    normalize : bool, default=True
        If ``False``, return the number of correctly classified samples.
        Otherwise, return the fraction of correctly classified samples.

    weights : array-like of shape (n_samples,), default=None
        Sample weights.
    
    multi_output : {"uniform_average", "weighted", "raw_values"}, default="uniform_average"
        Defines how to aggregate scores for multi-output data:
        
        - "uniform_average" : Average scores across all outputs (default)
        - "weighted"        : Weighted average using `weights`
        - "raw_values"      : Return scores for each output separately

    Returns
    -------
    score : float
        If ``normalize == True``, returns the fraction of correctly classified samples,
        else returns the number of correctly classified samples.

        The best performance is 1.0 with ``normalize == True`` and the number
        of samples with ``normalize == False``.


    Examples
    --------
    >>> from rslearn.metrics import accuracy_score
    >>> y_pred = [0, 2, 1, 3]
    >>> y_true = [0, 1, 2, 3]
    >>> accuracy_score(y_true, y_pred)
    0.5
    >>> accuracy_score(y_true, y_pred, normalize=False)
    2.0

    In the multilabel case with binary label indicators:

    >>> import numpy as np
    >>> accuracy_score(np.array([[0, 1], [1, 1]]), np.ones((2, 2)))
    0.75
    """

    check_multioutput(parameter=multi_output)

    # Converting to Arrays
    y_true, y_pred = convert_array(y_true, y_pred)

    # Checking for Array Size MisMatch
    shape_checker(y_true, y_pred, output_mode=True)
    
    
    # Handling Single Output Metrics
    if dim_validator(y_true):
        return _accuracy_score_helper_1d(y_true=y_true, y_pred=y_pred, normalize=normalize)

    
    outputs = _accuracy_score_helper_2d(y_true=y_true, y_pred=y_pred, normalize=normalize)

    # Multioutput Case Handler
    return multi_output_selector(multi_output_param=multi_output, scores=outputs, weights=weights)


        
"""Accuracy score Helper for 1D"""
def _accuracy_score_helper_1d(y_true, y_pred, normalize=True):
    y_true, y_pred = convert1D(y_true, y_pred)
    
    correct_count = 0

    length = len(y_true)

    for true, pred in zip(y_true, y_pred):
        if true == pred:
            correct_count += 1

    # Returning Accuracy
    if normalize is False:
        return correct_count
    
    return correct_count/length

"""Accuracy score Helper for 2D sparse metrics"""
def _accuracy_score_helper_2d(y_true, y_pred, normalize=True):
    n_col = y_true.shape[1]
    
    output = []
    for col in range(n_col):
        selected_true = y_true[:, col]
        selected_pred = y_pred[:, col]

        accuracy = _accuracy_score_helper_1d(selected_true, selected_pred, normalize=normalize)
        output.append(accuracy)
    
    return np.array(output)

def confusion_metrics(y_true, y_pred):
    """
    Confusion Metrics  

    Metrics of `TP`, `FP`, `TN`, `FN` combination for visible output for `Recall` and `precision` and `f1_score`  

    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) labels. Multi class also supported.    

    y_pred : 1d array-like, or label indicator array / sparse matrix
        Predicted labels, as returned by a classifier.  
    
    Returns
    -------
    (n_class, n_class) `np.array` metrics  

    NOTE: y_true & y_pred length must be same.  

    Raise
    -----
    Invalid Length Error.  

    Example
    -------
    >>> from rslearn.metrics import confusion_metrics
    >>> cm = confusion_metrics(true, pred) # returned metrics
    >>> print(cm)

    """
    y_true, y_pred = convert_array(y_true, y_pred)

    y_true, y_pred = convert1D(y_true, y_pred) # converting shape (n, 1) to (n,)

    shape_checker(y_true, y_pred, output_mode=True) # checking length to raise more & more errors :)


    classes = np.unique(np.concatenate((y_true, y_pred))) # seprating class
    num_class = len(classes)

    class_to_index = {c: i for i, c in enumerate(classes)}


    cm = np.zeros((num_class, num_class), dtype=int)

    for true, pred in zip(y_true, y_pred): # updating metrics
        cm[class_to_index[true], class_to_index[pred]] += 1

    return cm


def cm_helper(cm):
    n_classes = cm.shape[0]
    total = np.sum(cm)

    TP = np.diag(cm)
    FP = np.sum(cm, axis=0) - TP
    FN = np.sum(cm, axis=1) - TP
    TN = total - (TP + FP + FN)

    return TP, FP, FN, TN


def precision(y_true, y_pred):
    """
    precision  

    Quality Measurement of Model for positve Values   

    `Formula`
    ---------
    precision = TP / (TP + FP)

    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) labels. Multi class also supported.    

    y_pred : 1d array-like, or label indicator array / sparse matrix
        Predicted labels, as returned by a classifier.  
    
    Returns
    -------
    float64 

    NOTE: y_true & y_pred length must be same.  

    Raise
    -----
    Invalid Length Error.  

    Example
    -------
    >>> from rslearn.metrics import precision
    >>> score = precision(true, pred) # returned float value
    >>> print(score)

    """
    cm = confusion_metrics(y_true, y_pred)
    TP, FP, _, _ = cm_helper(cm)

    return TP / (TP + FP)

def recall(y_true, y_pred):
    """
    precision  

    Quality Measurement of Model for find all positve Values   

    `Formula`
    ---------
    precision = TP / (TP + FN)

    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) labels. Multi class also supported.    

    y_pred : 1d array-like, or label indicator array / sparse matrix
        Predicted labels, as returned by a classifier.  
    
    Returns
    -------
    float64 

    NOTE: y_true & y_pred length must be same.  

    Raise
    -----
    Invalid Length Error.  

    Example
    -------
    >>> from rslearn.metrics import recall
    >>> score = recall(true, pred) # returned float value
    >>> print(score)
    """

    cm = confusion_metrics(y_true, y_pred)
    TP, _ , FN, _ = cm_helper(cm)

    return TP / (TP + FN)

def f1_score(y_true, y_pred):

    """
    F1 Score    

    The F1 score is a balanced metric for classification, particularly useful
    for imbalanced datasets. It reaches its best value at 1 and worst at 0.  

    `Formula`
    ---------
    F1 = 2 * precision * recall / (precision + recall + epsolon)

    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) labels. Multi class also supported.    

    y_pred : 1d array-like, or label indicator array / sparse matrix
        Predicted labels, as returned by a classifier.  
    
    Returns
    -------
    float64 

    NOTE: y_true & y_pred length must be same.  

    Raise
    -----
    Invalid Length Error.  

    Example
    -------
    >>> from rslearn.metrics import f1_score
    >>> score = f1_score(true, pred) # returned float value
    >>> print(score)
    """

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)

    return 2 * (p * r) / (p + r + 1e-9)
