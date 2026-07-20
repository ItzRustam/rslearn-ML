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
Logistic Regression implementation using Gradient Descent.

Notes
-----
- This implementation uses gradient descent for optimization.
- Feature scaling is highly recommended for better convergence and performance.

Recommended preprocessing:
    from rslearn.preprocessing import StandardScaler, MinMaxScaler

Example:
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

- Without scaling, the model may converge slowly or produce suboptimal results.
"""

import numpy as np
from rslearn.BaseEstimators import _base
from rslearn.preprocessing import StandardScaler
from rslearn.metrics import (accuracy_score,
                            f1_score, 
                            recall, 
                            precision,
                            )
from rslearn.Errors import *
from rslearn.BaseEstimators import BaseEstimator
from typing import List

class LogisticRegression(BaseEstimator):

    """
    Logistic Regression
    -------------------
    Logistic Regression for 1D and 2D metrics both with binary and Catogirical classification support  

    use any Scaler for better result and accuracy specially in catogirical classification  

    Parameters
    ----------

    solver: liblinear/solver/auto
        liblinear for Binary Classification  
        saga for Catogirical Classification
        auto for automaticly chooses (Default)  
    
    lr: learning_rate  
        Default `0.01`
    
    Methords
    --------
    fit: function for fitting the model  
        Parameters in function doc string
    
    predict: function for prediction after fitting  

    Example
    -------
    >>> from rslearn.linear_model import LogisticRegression
    >>> Model = LogisticRegression()
    >>> Model.fit(X, y)
    >>> pred = Model.predict(X_test)


    """

    def __init__(self, solver="auto", lr = 0.001, weights : np.array= None, bias : float = None, catogirical_model : List[LogisticRegression] = [], max_itr : int =3000, hard_scale_off = False):
        if solver not in ["saga", "liblinear", "auto"]:
            raise InvalidValueError(f"Solver Must be saga or liblinear or auto (Default), Got {solver}")

        super().__init__(lr, max_itr, weights, bias)

        self.solver = solver
        self.type = "classification" # Flag for Pipeline
        self._model = "LogisticRegression"
        self._cato_model = catogirical_model
        self.hard_scale_off = hard_scale_off

    # Probablity predictor for catogirical classification
    def predict_proba(self, X):
        X = np.asarray(X)
    
        if self.solver == "liblinear":
            z = X @ self.weights + self.bias
            probs_1 = 1 / (1 + np.exp(-z))
            probs_0 = 1 - probs_1
            return np.vstack((probs_0, probs_1)).T
    
        else:
            probs = [m.predict_proba(X)[:, 1] for m in self._cato_model]
            probs = np.vstack(probs).T
    
            # normalize
            probs = probs / probs.sum(axis=1, keepdims=True)
            return probs

    def fit(self, X, y, scale=True):

        """
        Function for fitting Logistic Regression Model

        Parameters
        ----------
        X: feature set for model training
            2D or 1D metrics | `np.array`, `DataFrame`  
        
        y: correct value for X features set
            1D array | `np.array`  

        scale: Auto Scales Data On StandardScaler if True else Not
            Default `True`
        
        Returns
        -------
        None
        """
        self.flag = False # Validation to be False.
        X = np.asarray(X)
        y = np.asarray(y)
        y = y.reshape(-1)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        _base.shape_checker(X, y, output_mode=False) # Checking Shapes of Arrays

        if self.hard_scale_off:
            pass
        elif scale:
            X = self._scale_True(X, scaled=False)
            self.flag = True
        else:
            X = self._scale_False(X, scaled=False) # Gradient Stability
            self.flag = False
        self.fitted_shape=X.shape


        # Handling solvers in auto mode
        if self.solver == "auto":
            unique = np.unique(y)
            if len(unique) == 2:
                self.solver = "liblinear"
            else:
                self.solver = "saga"

        # Diffrent condition for fit
        if self.solver == "liblinear":
            Model = _binary_fit(X=X, y=y, lr=self.lr, max_itr=self.max_itr)
            self.weights, self.bias = Model.fit(weights=self.weights, bias=self.bias)
        
        else:
            Model = _catogirical_fit(X=X, y=y, max_itr=self.max_itr)
            Model.fit()
            self._cato_model = Model.models # Saving saga Model
            self.weights = np.array([0., 0.])
            self.bias = 0
        
        self._fitted = True

    def predict(self, X):

        """
        Function for predict for Logistic Regression

        Parameter
        --------
        X: new Data for prediction  
            n_sample, n_features of X should be same as data on which model trained  
            preferd `np.array`, `DataFrame`
        """

        if len(X) == 0:
            raise InvalidValueError("Got Empty Array")
        
        if not self._fitted:
            raise InvalidValueError("Model has not been fitted yet.")

        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if self.fitted_shape[1] != X.shape[1]:
            raise InvalidValueError(f"Invalid Shape, Model trained on {self.fitted_shape} but got {X.shape}")

        if self.hard_scale_off:
            pass
        # Scaling If Available when hard_scale_off is False
        if self.flag:
            X = self._scale_True(X, scaled=True)

        # else Gradient Stability opration with X/max(X) + 1e-9
        else:
            X = self._scale_False(X, scaled=True)
        

        probs = self.predict_proba(X)

        if probs.shape[1] == 2:
            return (probs[:, 1] >= 0.5).astype(int)
            
        else:
            return np.argmax(probs, axis=1)
    
    def evaluate(
        self,
        X=None,
        y_pred=None,
        y_true=None
    ):
        
        """
        Evaluate model performance using various classification metrics.

        Parameters:
            X: array-like of shape (n_samples, n_features), default=None
                Input data to evaluate predictions on. If provided, model will generate
                predictions and use them for evaluation.
            y_pred: array-like of shape (n_samples,), default=None
                Predictions to use for evaluation. Only one of X or y_pred should be provided.
            y_true: array-like of shape (n_samples,), default=None
                True target values for evaluation.
        """


        return super()._eval(X=X, y_pred=y_pred, y_true=y_true)
    
    def save(self, file_name="rslearn_model.rsl"):
        super().save(file_name=file_name, solver=self.solver, catogirical_models=self._cato_model)
        
                

# For liblinear (Default)
class _binary_fit:
    def __init__(self,X , y, lr, max_itr : int = 1000):
        self.lr = lr
        self.X = X
        self.y = y
        self.max_itr = max_itr

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, weights=None, bias=None):
        X = self.X
        y = self.y
        n_rows, n_features = X.shape
        if weights is None:
            weights = np.random.uniform(0.2, 3, n_features)
        if bias is None:
            bias = 0

        for _ in range(self.max_itr):
            z = np.dot(X, weights) + bias 
            y_pred = self._sigmoid(z)

            # Gradients
            pos_weight = (n_rows / (2 * np.sum(y)))      # weight for class 1
            neg_weight = (n_rows / (2 * np.sum(1 - y)))  # weight for class 0

            weights_factor = y * pos_weight + (1 - y) * neg_weight

            dw = (1/n_rows) * np.dot(X.T, (weights_factor * (y_pred - y)))
            db = (1/n_rows) * np.sum(weights_factor * (y_pred - y))

            # update
            weights -= self.lr * dw
            bias -= self.lr * db

        return weights, bias
            

# For Catogrical
class _catogirical_fit:
    def __init__(self, X, y, max_itr=1000):
        self.X = X
        self.y = y
        self.max_itr=max_itr
    
    def fit(self):
        X = self.X
        y = self.y
        self.models = []
        self.classes = np.unique(y)

        for c in self.classes:
            model = LogisticRegression(solver="liblinear", max_itr=self.max_itr)
            y_bin = (y == c).astype(int)
            model.fit(X, y_bin)
            self.models.append(model)

    def predict(self, X):
        probs = [m.predict_proba(X) for m in self.models]
        probs = np.vstack(probs).T
        return np.argmax(probs, axis=1)
