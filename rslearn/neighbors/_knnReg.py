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
from rslearn.Errors import *
from rslearn.preprocessing import StandardScaler
from rslearn.metrics import EuclidienDisctance
from rslearn.metrics import (
    r2_score,
    mse,
    mae,
    rmse
)
from rslearn.BaseEstimators import _base
from rslearn.BaseEstimators import BaseEstimatorKNN



class KNNRegressor(BaseEstimatorKNN):
    def __init__(self, k_neighbors=5, hard_scale_off : float = False):
        """
        KNNRegressor Class

        A K-Nearest Neighbors Regressor for regression tasks. This implementation supports optional data scaling 
        and provides methods to fit the model, generate predictions, and evaluate performance.

        Parameters:  
            k_neighbors: int  
                Number of neighbors to consider for regression, default = 5  
 
        Methods:  
            fit(X, y, scale=True)
                Fit the model using X and target y. Can optionally perform scaling of features.
 
            predict(X_new)  
                Generate predictions for new input data X_new based on fitted model.

            evaluate(X=None, y_pred=None, y_true=None)
                Evaluate model performance using various regression metrics.
        """

        super().__init__(k_neighbors=k_neighbors)
        self.type = "regression"
        self._model = "KNNRegressor"
        self.hard_scale_off = hard_scale_off
    
    def fit(self, X, y, scale=True):
        """
        Fit the model using the provided training data.

        Parameters:
            X: array-like of shape (n_samples, n_features)
                Training data.
            y: array-like of shape (n_samples,)
                Target values for each sample during training.
            scale: bool, default=True
                Whether to perform scaling on the input features before fitting.
        """
        X = np.asarray(X)
        y = np.asarray(y)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if len(X) != len(y):
            raise LengthError(f"Length Mismatch {(len(X), len(y))}")
        
        if self.hard_scale_off:
            pass
        elif scale:
            X = self._scale_True(X, scaled=False)
            self.flag = True
        else:
            X = self._scale_False(X, scaled=False)
        

        self.fitted_x = X
        self.fitted_y = y

        self.fitted_shape = X.shape
        self._fitted = True
    
    def predict(self, X_new):
        """
        Generate predictions for new input data using the fitted model.

        Parameters:
            X_new: array-like of shape (n_samples, n_features)
                Input data to make predictions on.
        """
        
        if not(self._fitted):
            raise NotFittedError("Not has not been fitted yet.")
        
        X_new = np.asarray(X_new)
        if X_new.ndim == 1:
            X_new  = X_new.reshape(-1, 1)

        if X_new.shape[1] != self.fitted_shape[1]:
            raise InvalidShape(f"Invalid Shape, Model fitted on {self.fitted_shape} Got {X_new.shape}")
        
        if self.hard_scale_off:
            pass
        elif self.flag:
            X_new = self._scale_True(X_new, scaled=True)
        else:
            X_new = self._scale_False(X_new, scaled=True)
        
        response = []

        for idx in range(len(X_new)):

            current_sample  = X_new[idx]

            
            distance = EuclidienDisctance(self.fitted_x, current_sample)

            indices = np.argsort(distance)[:self.k]

            votes = self.fitted_y[indices]

            val = np.mean(votes)
            response.append(val)
        
        return np.array(response)

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
    

if __name__ == '__main__':
    data = np.array([
        [1,2],
        [3, 4],
        [4, 5]
    ])

    key = np.array([2])
    model = KNNRegressor()
    model.fit(data, key)