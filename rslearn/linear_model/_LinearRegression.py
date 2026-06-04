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
Things : - 

# it is linear regression
y = m1x1 + m1x2 + m3x3 + ... + MnXn + b

y = prediction
m = weight
x = value
b = bias

loss = prediction - real_val
dw = gradient descent of weight
db = gradient descenf of bias

It uses Gradients so, Use `StandardScaler` or `MinMaxScaler` for better result  

Scalers...
>>> from rslearn.preprocessing import StandardScaler, MinMaxScaler
Read READNE.md or Documentation for More Information about their Functions
"""

from rslearn.BaseEstimators import _base
from rslearn.preprocessing import StandardScaler
from rslearn.metrics import (
                            r2_score, 
                            mse,
                            rmse,
                            mae
                            )
import numpy as np


class LinearRegression():


    def __init__(self, regulization=None, alpha : float = 0.1, l1_ratio=0.5):

        """
        Linear Regression
        ------------------------

        linear Regression for 1D and 2D metrics arrays  using gradient descents and regulization
        use Scalers like MinMaxScaler or StandardScaler before fitting for Handle large value

        Example
        --------
        regulization: regulizing option to avoid overfitting
            options:  `l1` for Lasso  
                      `l2` for Ridge  
                      `elastic_net` for elastic_net
            
            Default: None, For No regulization.
        
        alpha: alpha value for Ridge, Lasso, ElasticNet  
            Default: 0.1  
        
        l1_ratio: Lasso Ratio for Better ElasticNet Gradient and MSE  
            Default: 0.5  
        
        Functions
        ---------
        fit()
            Function for Train Model | use MinMaxScaler for good Computation and Prediction  
            Parameters - given in Function
        
        get_weight_bias()
            Returns Selected weight and Bias for minimum loss  
        
        predict()
            Prediction generator from Model  
        
        
        Example
        -------

        >>> from rslearn.linear_model import LinearRegression
        >>> Model = LinearRegression()
        >>> X = np.array([10, 20, 30]) # List also works.
        >>> y = np.array([5, 10, 15])
        >>> Model.fit(X, y, scale=True) # You can change learning_rate too
        >>> print(f"Weight & Bias: {Model.get_weight_bias()}")
        >>> prediction = Model.predict(np.array([40, 50]))
        >>> print(f"Prediction: {prediction}")
            

        
        """

        self.weights = None
        self.bias = None
        self.Scaler = StandardScaler() # Scaler
        self.flag = False # Flag For Scaler Status
        self.type = "regression"
        self.fitted_shape = None # Edge Case
        self._fitted = False # Edge Case

        valid_params = {"l1", "l2", "elastic_net", None}
        if regulization not in valid_params:
            raise ValueError(f"regulization parameter is not supported, supported Parameters {valid_params}")
        
        self.calculate_error = self._regulizing_linear_helper(regulization=regulization, alpha=alpha, l1_ratio=l1_ratio)
            
    


    def fit(self,
            X_arr ,
            y_arr , 
            weights= None,
            bias = None,
            learning_rate : float = 0.001,
            min_loss : float = 0.1,
            max_itr : int = 18000,
            verbose : bool = False,
            scale : bool = True
            ):
        """

        Input Param*
        __________
        X = Data to Train 1D or 2D array, Dtype = np.array   

        Y = True value a.k.a. original prediction 1D or 2D array, Dtype = np.array  

        max_itr = loop to update weight and bias, Dtype = int and default = 18000  | No Input need  

        learning_rate = how fast weights should update, Dtype = float, Default = 0.01

        scale: Auto Scales Data On StandardScaler if True else Not
            Default `True`

        weights = enter custom weight |  optional  

        bias = enter custom bias |  optional

        min_loss = minimum loss where to stop the loop Default = 0.2, and it's almost best for gradient descent

        verbose = True/False | Prints The iteration where Model Fitted Successfully

        -----------------

        Change the `learning_rate` or Use `Scalers` if output or weights contains 'e' e.g -1.8038873e+163
        """

        X, y = _base.convert_array(arr1=X_arr, arr2=y_arr) # Converting to np.array
        y = y.reshape(-1)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        _base.shape_checker(X, y, output_mode=False)

        if scale:
            X = self.Scaler.fit_transform(X)
            self.flag = True
        else:
            X = X/max(X)
            self.maxx = max(X)
        

        
        n_samples, n_feature = X.shape
        self.fitted_shape = X.shape

        np.random.seed(7)
        if weights is None:
            weights = np.random.uniform(0.2, 3, n_feature)
        
        if bias is None:
            bias = 0

        iteration  = 0

        while iteration < max_itr:
            pred = np.dot(X, weights) + bias # prediction

            mse_error : float = self.calculate_error.get_error(y_true=y, y_pred=pred, weights=weights)

            if mse_error <= min_loss:
                if verbose:
                    print(f"Model Succesfully Fitted at #{iteration} iteration")
                break

            loss = pred - y # Loss for Gradients
            dw = (2/n_samples) * np.dot(X.T, loss) + self.calculate_error.get_weight_gradient(weights=weights)
            db = (2/n_samples) * np.sum(loss)

            weights -= learning_rate * dw
            bias -= learning_rate * db

            if _base.check_overflow(weights=weights, bias=bias):
                print("NaN detected, stopping training, Use Scalers to avoid it")
                break

            iteration += 1


        
        self.weights = weights
        self.bias = bias
        self._fitted = True

    def get_weight_bias(self) -> np.array:
        """Input = None, 
        O/P - (np.array, float64)
        >>> weights, bias = Model.get_weight_bias()
        """

        return (self.weights, self.bias)
    
    def predict(self, new_data : np.array) -> np.array:
        """
        Input Format = 1D or 2D np.array
        Output Format = 1D np.array
        """
        if len(new_data) == 0:
            raise ValueError("Got Empty Array")

        if not self._fitted:
            raise ValueError("Model has not been fitted yet.")
        
        new_data = np.asarray(new_data, dtype=float)
        if new_data.ndim == 1:
            new_data = new_data.reshape(-1, 1)

        if self.fitted_shape[1] != new_data.shape[1]:
            raise ValueError(f"Invalid Shape, Model trained on {self.fitted_shape} but got {new_data.shape}")

        if self.flag:
            new_data = self.Scaler.transform(new_data)
        else:
            new_data = new_data/self.maxx # Stability
        

        return (np.dot(new_data, self.weights) + self.bias).round()

    def evaluate(
        self,
        X=None,
        y_pred=None,
        y_true=None
    ):
        
        """
        `analysis` Method

        Function to Evaluate All suitable Metrics Algorithams and print Them 

        Parameters
        ----------
        y_pred: predictions from Model

        y_true: Correct Values to Evaluate

        Returns
        -------
        None
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
        y_pred = np.asarray(y_pred, dtype=float) # if y_pred != None, Otherwise Model will return `np.array``
        y_true = np.asarray(y_true, dtype=float)
        y_true = y_true.reshape(-1) # reshaping y_true to 1D to match with y_pred

        _base.shape_checker(arr1=y_true, arr2=y_pred, output_mode=True)

        # Evaluations for Regression Task
        r2_Score = r2_score(y_true=y_true, y_pred=y_pred)
        MSE = mse(y_true=y_true, y_pred=y_pred)
        MAE = mae(y_true=y_true, y_pred=y_pred)
        RMSE = rmse(y_true=y_true, y_pred=y_pred)

        evaluations = { # Storing in Dict
            "r2_score": r2_Score,
            "mse": float(MSE),
            "mae": float(MAE),
            "rmse": float(RMSE)
        }

        # Returning `prediction` and `Evaluation` for future Flask/FastAPI support
        return {
            "prediction" : y_pred,
            "evaluation" : evaluations
        }
        


        

    class _regulizing_linear_helper:
        def __init__(self, alpha=0.1, regulization=None, l1_ratio = 0.5):
            self.alpha = alpha
            self.regulization = regulization
            self.l1_ratio = l1_ratio
        
        def get_error(self, y_true, y_pred, weights):
            mse_error = mse(y_true, y_pred)
            if self.regulization is None:
                return mse_error
            
            if self.regulization == "l1":
                reg = self.alpha * np.sum(np.abs(weights))
                return  mse_error + reg
            
            if self.regulization == "l2":
                reg = self.alpha * np.sum(np.square(weights))
                return mse_error + reg
            
            if self.regulization == "elastic_net":
                l1 = self.alpha * self.l1_ratio
                l2 = self.alpha * (1 - self.l1_ratio)

                reg = l1 * np.sum(np.abs(weights)) + l2 * np.sum(np.square(weights))
                return mse_error + reg
        
        def get_weight_gradient(self, weights):
            if self.regulization == "l1":
                return self.alpha * np.sign(weights)
            
            if self.regulization == "l2":
                return 2 * self.alpha * weights
            
            if self.regulization == "elastic_net":
                l1 = self.alpha * self.l1_ratio
                l2 = self.alpha * (1 - self.l1_ratio)

                return l1 * np.sign(weights) + 2 * l2 * weights

            return 0





    

if __name__ == "__main__":
    Model = LinearRegression()