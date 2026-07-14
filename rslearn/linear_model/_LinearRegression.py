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

# Linear Regression Implementation
"""
y = m1x1 + m2x2 + ... + MnXn + b
y = prediction
m = weight
x = value
b = bias
loss = prediction - real_val
dw = gradient descent of weight
db = gradient descenf of bias

It utilizes Gradients, so it's recommended to use `StandardScaler` or `MinMaxScaler` for better performance.

Scalers...
>>> from rslearn.preprocessing import StandardScaler, MinMaxScaler
For more information, refer to README.md or the documentation.

Author: ItzRustam
"""

from rslearn.BaseEstimators import _base
from rslearn.BaseEstimators import BaseEstimator
import numpy as np
from rslearn.Errors import *
from rslearn.metrics import mse


class LinearRegression(BaseEstimator):


    def __init__(self, regulization=None, alpha : float = 0.1, l1_ratio=0.5, lr : float = 0.001, max_itr : int =1000, weights : np.array = None, bias : float = None, min_loss : float=0.1):

        """
        Linear Regression
        ------------------------

        linear Regression for 1D and 2D metrics arrays  using gradient descents and regulization
        use Scalers like MinMaxScaler or StandardScaler before fitting for Handle large value.

        Parameters
        --------
        regulization: regulizing option to avoid overfitting
            options:  `l1` for Lasso  
                      `l2` for Ridge  
                      `elastic_net` for elastic_net
            
            Default: None, For No regulization.
        
        alpha : float, default: 0.1 
            alpha value for Ridge, Lasso, ElasticNet  
             
        
        l1_ratio : float, default: 0.5 
            Lasso Ratio for Better ElasticNet Gradient and MSE    
             
        
        lr : float, default=0.001
            Learning rate for weight updates.  

        max_itr : int, default=1000
            Maximum number of iterations for the gradient descent algorithm.  

        weights : np.array, optional
            Initial weights for the model. If None, weights are initialized randomly.  

        bias : float, optional
            Initial bias for the model. If None, bias is initialized to 0.  

        min_loss : float, default=0.1
            Minimum loss threshold to stop training.  

               
        Methods
        ---------
        fit()
            Method for Train Model  
        
        get_weight_bias()
            Returns Selected weight and Bias for minimum loss    
        
        predict()
            Prediction generator from Model    
        
        Example
        -------
        >>> from rslearn.linear_model import LinearRegression
        >>> Model = LinearRegression(max_itr=1800)
        >>> X = np.array([10, 20, 30]) # List also works.
        >>> y = np.array([5, 10, 15])
        >>> Model.fit(X, y, scale=True)
        >>> print(f"Weight & Bias: {Model.get_weight_bias()}")
        >>> prediction = Model.predict(np.array([40, 50]))
        >>> print(f"Evaluations: {Model.evaluate(y_pred=prediction, y_true=[20, 25])}")
        """

        super().__init__(lr, max_itr, weights, bias)

        valid_params = {"l1", "l2", "elastic_net", None}
        if regulization not in valid_params:
            raise InvalidValueError(f"regulization parameter is not supported, supported Parameters {valid_params}")
        
        self.calculate_error = self._regulizing_linear_helper(regulization=regulization, alpha=alpha, l1_ratio=l1_ratio)
        self.min_loss = min_loss
        self._model = "LinearRegression"
        self.type = "regression"
    


    def fit(self,
            X_arr ,
            y_arr , 
            scale : bool = True
            ):
        """
        `Fit` the linear regression model to the given data.

        Parameters
        ----------
        X_arr : np.array
            Feature matrix of shape (n_samples, n_features).

        y_arr : np.array
            Target vector of shape (n_samples,).

        scale : bool, default=True
            Whether to scale the features using StandardScaler before fitting. If True,
            features are scaled; otherwise, they are not scaled.

        Returns
        -------
        str
            A success message indicating that the model has been fitted successfully.
        
        Notes
        -----
        This method performs gradient descent optimization to find the optimal weights
        and bias for minimizing the mean squared error (MSE). If regularization is applied,
        it adjusts the loss function accordingly.
        """


        X, y = _base.convert_array(arr1=X_arr, arr2=y_arr) # Converting to np.array
        y = y.reshape(-1)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        _base.shape_checker(X, y, output_mode=False)

        if scale:
            X = self._scale_True(X, scaled=False)
            self.flag = True
        else:
            X = self._scale_False(X, scaled=False)
        

        
        n_samples, n_feature = X.shape
        self.fitted_shape = X.shape

        np.random.seed(7)
        if self.weights is None:
            weights_temp = np.random.uniform(0.2, 3, n_feature)
        else:
            weights_temp = self.weights
        
        if self.bias is None:
            bias_temp = 0
        else:
            bias_temp = self.bias

        iteration  = 0

        while iteration < self.max_itr:
            pred = np.dot(X, weights_temp) + bias_temp # prediction

            mse_error : float = self.calculate_error.get_error(y_true=y, y_pred=pred, weights=weights_temp)

            if mse_error <= self.min_loss:
                break

            loss = pred - y # Loss for Gradients
            dw = (2/n_samples) * np.dot(X.T, loss) + self.calculate_error.get_weight_gradient(weights=weights_temp)
            db = (2/n_samples) * np.sum(loss)

            weights_temp -= self.lr * dw
            bias_temp -= self.lr * db

            if _base.check_overflow(weights=weights_temp, bias=bias_temp):
                print("NaN detected, stopping training, Use Scalers to avoid it")
                break

            iteration += 1


        
        self.weights = weights_temp
        self.bias = bias_temp
        self._fitted = True
        
        return "Model Fitted Successfully"

    
    def predict(self, new_data : np.array) -> np.array:
        """
        Input Format = 1D or 2D np.array
        Output Format = 1D np.array
        """
        if len(new_data) == 0:
            raise InvalidValueError("Got Empty Array")

        if not self._fitted:
            raise InvalidValueError("Model has not been fitted yet.")
        
        new_data = np.asarray(new_data, dtype=float)
        if new_data.ndim == 1:
            new_data = new_data.reshape(-1, 1)

        if self.fitted_shape[1] != new_data.shape[1]:
            raise InvalidValueError(f"Invalid Shape, Model trained on {self.fitted_shape} but got {new_data.shape}")

        if self.flag:
            new_data = self._scale_True(X=new_data, scaled=True)
        else:
            new_data = self._scale_False(X=new_data, scaled=True)
        

        return (np.dot(new_data, self.weights) + self.bias).round()

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