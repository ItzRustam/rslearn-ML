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
    f1_score,
    accuracy_score,
    recall,
    precision
)
from rslearn.BaseEstimators import _base



class KNNClassifier:
    def __init__(self, k_neighbors=5):
        """
        KNNClassifier Class

        A K-Nearest Neighbors classifier for classification tasks. This implementation supports optional data scaling 
        and provides methods to fit the model, generate predictions, and evaluate performance.

        Parameters:  
            k_neighbors: int  
                Number of neighbors to consider for classification, default = 5  
 
        Methods:  
            fit(X, y, scale=True)
                Fit the model using X and target y. Can optionally perform scaling of features.
 
            predict(X_new)  
                Generate predictions for new input data X_new based on fitted model.

            evaluate(X=None, y_pred=None, y_true=None)
                Evaluate model performance using various classification metrics.
        """
        self.k = k_neighbors
        self.Scaler = StandardScaler() # Scaler
        self.flag = False # Flag For Scaler Status
        self.type = "classification"
        self.fitted_shape = None # Edge Case
        self._fitted = False # Edge Case
        self.fitted_x = None
        self.fitted_y = None
    
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
        
        if scale:
            X = self.Scaler.fit_transform(X)
            self.flag = True
        else:
            X = X/np.max(X)
        

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
        
        if self.flag:
            X_new = self.Scaler.transform(X_new)
        else:
            X_new = X_new/np.max(X_new)
        
        response = []

        for idx in range(len(X_new)):

            current_sample  = X_new[idx]

            
            distance = EuclidienDisctance(self.fitted_x, current_sample)

            indices = np.argsort(distance)[:self.k]

            votes = self.fitted_y[indices]

            unique, counts = np.unique(votes, return_counts=True)
            most_frequent_value = unique[np.argmax(counts)]
            response.append(most_frequent_value)
        
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


        if not self._fitted: # If Model is not fitted
            raise RuntimeError(
                "This model is not trained yet. Call 'fit()' before using 'evaluate()'."
            )

        if y_true is None: # Edge case : Nothing to compare
            raise ValueError("Invalid Arguments `y_true` `None`")
        
        
        if y_pred is None:
            if X is None: # Edge case: Both `X` and `y_pred` are None
                raise ValueError("parameter `X` and `y_pred` Both given None.")
        
            y_pred = self.predict(X) # Getting Prediction


        # Converting to `np.array`` if they are not
        y_pred = np.asarray(y_pred) # if y_pred != None, Otherwise Model will return `np.array``
        y_true = np.asarray(y_true)
        y_true = y_true.reshape(-1) # reshaping y_true to 1D to match with y_pred

        _base.shape_checker(arr1=y_true, arr2=y_pred, output_mode=True)

        # Evaluations for Classification Tasks Task
        accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
        F1 = f1_score(y_true=y_true, y_pred=y_pred)
        Recall = recall(y_true=y_true, y_pred=y_pred)
        Precision = precision(y_true=y_true, y_pred=y_pred)

        evaluations = { # Storing in Dict
            "accuracy_score": float(accuracy),
            "f1_score": F1,
            "recall": Recall,
            "precision": Precision
        }

        # Returning `prediction` and `Evaluation` for future Flask/FastAPI support
        return {
            "prediction" : y_pred,
            "evaluation" : evaluations
        }
    

if __name__ == '__main__':
    data = np.array([
        [1,2],
        [3, 4],
        [4, 5]
    ])

    key = np.array([2])
    model = KNNClassifier()
    model.fit(data, key)