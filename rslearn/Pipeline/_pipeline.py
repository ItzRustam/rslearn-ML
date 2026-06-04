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
rslearn.Pipeline

`_pipeline.py`   

This File Contains Pipeline for rslearn

Features
--------

* Auto Scaling
* Auto Splitting with `validation_split = True`
* Inbuilt Analysis Tool

"""

import numpy as np 
from rslearn.model_selection import train_test_split
from rslearn.metrics import (accuracy_score,
                            f1_score, 
                            recall, 
                            precision,
                            r2_score, 
                            mse,
                            rmse,
                            mae
                            )
from pprint import pprint

class pipeline:
    def __init__(self, params={}, validation_split=False, split_params={"test_size": 0.25, "random_state": 67, "stratify": None},):
        """
        pipeline

        Tool to Automate The Process of `Scaling`, `training` & `Analysis`

        Parameter
        ---------

        params: parameters to config `Model` & `Scaler`
            Default {}, Dtype = `dict`
            options:
                {
                "model" : LinearRegression(),
                "scaler": StandardScaler()
                }
        
        validation_split: Automatic Splitter & Test Evaluator
            Default - False  
            Recommanded - `True` For Great Experience
        
        split_params: Data Splitting Parameters
            Default: Given To Function  
            Only When `validation_split` = `True`
        
        Returns
        -------
        None

        Methods
        -------
        fit: Function to Train The Model
            Params In Function Doc String
        
        predict: Function to get predictions from Trained Model
            Params In Function Doc String
        
        Analysis: Function to Evaluate All Supported Metrics Tools For `prediction`
            Params In Function Doc String
        
        Raises
        ------
        ValueError: when Empty params,    
            No Model In Parameter,  
            Empty Metrics,  
            prediction without fitting Model,  
            Array Shape MisMatch  
        
        Example
        -------
        With `validation_split` = `True`
        >>> from rslearn.preprocessing import StandardScaler
        >>> from rslearn.linear_model import LinearRegression

        >>> line = pipeline(
        >>>    params={
        >>>        "model": LinearRegression(regulization="l1"),
        >>>        "scaler": StandardScaler()
        >>>    },
        >>>    validation_split=True,
        >>>    )

        >>>    X = [
        >>>        [10, 20], 
        >>>        [40, 50], 
        >>>        [15, 15], 
        >>>        [40, 35], 
        >>>        [12, 15], 
        >>>        [25, 15], 
        >>>        [13, 12], 
        >>>        [15, 10], 
        >>>        [15, 15]
        >>>    ]
        >>>    y = [
        >>>        [30],
        >>>        [90],
        >>>        [30],
        >>>        [75],
        >>>        [27],
        >>>        [40],
        >>>        [25],
        >>>        [25],
        >>>        [30]
        >>>    ]
        >>>    line.fit(X=X, y=y)

        With `validation_split` = `False`

        >>> from rslearn.preprocessing import StandardScaler
        >>> from rslearn.linear_model import LinearRegression
        >>> line = pipeline(
        >>>     params={
        >>>         "model": LinearRegression(regulization="l1"),
        >>>         "scaler": StandardScaler()
        >>>     },
        >>>     validation_split=True,
        >>>     )
        >>>     X = [
        >>>         [10, 20], 
        >>>         [40, 50], 
        >>>         [15, 15], 
        >>>         [40, 35], 
        >>>         [12, 15], 
        >>>         [25, 15], 
        >>>         [13, 12], 
        >>>         [15, 10], 
        >>>         [15, 15]
        >>>     ]
        >>>     y = [
        >>>         [30],
        >>>         [90],
        >>>         [30],
        >>>         [75],
        >>>         [27],
        >>>         [40],
        >>>         [25],
        >>>         [25],
        >>>         [30]
        >>>     ]
        >>>     line.fit(X=X, y=y)
        """
        if len(params) == 0:
            raise ValueError("Empty Parameter For Pipeline")

        if "model" not in params:
            raise ValueError("No Model Selected in Parameters")
        
        self.Model = params['model']
        self.scaling = False
        if "scaler" in params:
            self.Scaler = params['scaler']
            self.scaling = True
        
        self.split_params = split_params
        self.trained=False

        self.split = validation_split
    
    def _helper(self, X, y, split_params, split):
        if split is False:
            return np.asarray(X, dtype=float), '_' , np.asarray(y, dtype=float), '_'
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.split_params['test_size'], random_state=self.split_params['random_state'], stratify=self.split_params['stratify'])

        return X_train, X_test, y_train, y_test
        

        
    def fit(self,
            X=None,
            y=None,
            verbose=True
            ):
            """ 
            `fit` Method for pipeline

            Function to Fit The Model

            Parameters
            ----------
            X: NxM metrics of `np.array` conatins The Feature to Train Model
            
            y: Correct Output For X Metrics

            Returns
            -------
            None
            """
            if X is None or y is None:
                raise AttributeError("""Invalid Attribute `X` or `y`
                                     Note: Check X or y not to be None""")
            
            
            X_train, X_test, y_train, y_test = self._helper(X=X, y=y, split_params=self.split_params, split=self.split)
            
            y_train = y_train.reshape(-1)

            if self.split:
                y_test = y_test.reshape(-1)

            self.Model.fit(X_train, y_train, scale=self.scaling)
            self.trained=True

            if self.split:
                y_pred = self.Model.predict(X_test)
                output = self.analysis(y_pred=y_pred, y_true=y_test)

                if verbose is False:
                    return output, X_test

                print("Predictions For X_test")
                print("X_test: ", self.Scaler.inverse_transform(X_test))
                print("Prediction: ", output['prediction'])
                print("Metrics Evaluations\n")
                pprint(output['evaluation'])
                
                print("Output is Returned By output, X_test\n")
                return output, self.Scaler.inverse_transform(X_test)

                

    
    def predict(self,
        new_data=None
        ):

        """
        `predict` method for pipeline

        Function to get Predictions from Trained Model

        Parameter
        ---------
        new_data: NxM metrics of new_data as Same shape of X

        Returns
        -------
        `np.array` of predictions
        """
        if new_data is None:
                raise AttributeError("""Invalid Attribute `new_data`
                                     Note: Check `new_data` not to be None""")

        if len(new_data) == 0:
            raise ValueError("Got Empty Metrics!")
        
        if self.trained is False:
            raise ValueError("Model Has Not Been Fitted yet!")

        if self.scaling:
            new_data = self.Scaler.transform(new_data)
                
                
        pred = self.Model.predict(new_data)
        return pred
    
    def evaluate(
        self,
        X=None,
        y_pred=None,
        y_true=None
    ):
        
        """
        `evaluate` Method

        Function to Evaluate All suitable Metrics Algorithams and print Them 

        Parameters
        ----------
        y_pred: predictions from Model

        y_true: Correct Values to Evaluate

        Returns
        -------
        Dict - dictonary of `predictions` & `Evaluations` on that  
            Keys: `prediction`, `evaluation`  
        """

        if not self.trained:
            raise RuntimeError(
                "This model is not trained yet. Call 'fit()' before using 'evaluate()'."
            )

        return self.Model.evaluate(X=X, y_pred=y_pred, y_true=y_true)



if __name__ == "__main__":
    print("rslearn.Pipeline Main Directry")