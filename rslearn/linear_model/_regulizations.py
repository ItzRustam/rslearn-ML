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

from rslearn.linear_model import LinearRegression

"""
This File Contains regulizing algorithams to avoid overfitting though the model  

Algorithams Contains  
----------------------
- `Lasso`
- `Ridge`
- `ElasticNet`
"""

class Lasso:

    """
    Lasso l1 regulization for Avoid Overfitting though abs()  
    NOTE: It uses LinearRegression Internly So make Sure to Scale youre Data and enter False in Scale Parameter in `fit()`  

    Parameters
    -----------

    alpha: alpha value for Ridge, Lasso, ElasticNet  
            Default: 0.1  
        
    l1_ratio: Lasso Ratio for Better ElasticNet Gradient and MSE  
        Default: 0.5  

    Returns
    -------
    None

    Example
    -------

    >>> from rslearn.linear_model import Lasso
    >>> LassoR = Lasso() # Using Default Parameters
    >>> LassoR.fit(X, y, Scale=True) # Auto Scales Basicly for better performence use StandardScaler
    >>> LassoR.predict(X_new)

    """

    def __init__(self, alpha=0.1, l1_score=0.5):
        self.alpha = alpha
        self.l1_score = l1_score
        self.weights, self.bias = None
        self.model = None
        self.type = "regression"
    
    def fit(self, X, y, scale=True, min_loss : float = 0.1):

        """
        `fit()` Function For `Lasso` to Train The Model  

        Parameters
        ----------

        X: array-like, 1D or 2D metrics for train model  

        y: array-like, 1D or 2D metrics of Real Value for X features

        Scaler: option to Scale Automaticly
            Default: True, Use `StandardScaler` and select `False` For better results
        
        Returns
        -------
        None
        """

        model = LinearRegression(regulization="l1", alpha=self.alpha, l1_ratio=self.l1_score)
        
        model.fit(X, y, scale=scale, verbose=False, min_loss=min_loss)

        self.weights, self.bias = model.get_weight_bias()
        self.model = model
    
    def predict(self, X_new):
        prediction = self.model.predict(X_new)
        return prediction
    
    def get_weight_bias(self):
        return (self.weights, self.bias)

    def evaluate(self, X=None, y_pred=None, y_true=None):
        return self.model.evaluate(X=X, y_pred=y_pred, y_true=y_true)


class Ridge:
    """
    Ridge `l2` regulization for Avoid Overfitting though square()
    NOTE: It uses LinearRegression Internly So make Sure to Scale youre Data and enter False in Scale Parameter in `fit()`  

    Parameters
    -----------

    alpha: alpha value for Ridge, Lasso, ElasticNet  
            Default: 0.1  
        
    l1_ratio: Lasso Ratio for Better ElasticNet Gradient and MSE  
        Default: 0.5  

    Returns
    -------
    None

    Example
    -------

    >>> from rslearn.linear_model import Ridge
    >>> RidgeR = Ridge() # Using Default Parameters
    >>> RidgeR.fit(X, y, Scale=True) # Auto Scales Basicly for better performence use StandardScaler
    >>> RidgeR.predict(X_new)

    """
    

    def __init__(self, alpha=0.1, l1_score=0.5):
        self.alpha = alpha
        self.l1_score = l1_score
        self.weights, self.bias = None
        self.model = None
        self.type = "regression" # Flags for pipeline Analysis
    
    def fit(self, X, y, scale=True, min_loss : float = 0.1):
        """
        `fit()` Function For `Ridge` to Train The Model  

        Parameters
        ----------

        X: array-like, 1D or 2D metrics for train model  

        y: array-like, 1D or 2D metrics of Real Value for X features

        Scaler: option to Scale Automaticly
            Default: True, Use `StandardScaler` and select `False` For better results
        
        Returns
        -------
        None
        """


        model = LinearRegression(regulization="l2", alpha=self.alpha, l1_ratio=self.l1_score)


        
        model.fit(X, y, scale=scale, verbose=False, min_loss=min_loss)

        self.weights, self.bias = model.get_weight_bias()
        self.model = model
    
    def predict(self, X_new):
        prediction = self.model.predict(X_new)
        return prediction
    
    def get_weight_bias(self):
        return (self.weights, self.bias)
    
    def evaluate(self, X=None, y_pred=None, y_true=None):
        return self.model.evaluate(X=X, y_pred=y_pred, y_true=y_true)

class ElasticNet:
    """
    `ElasticNet` regulization for Avoid Overfitting by Combination of `l1`, `l2`  
    NOTE: It uses LinearRegression Internly So make Sure to Scale youre Data and enter False in Scale Parameter in `fit()`  

    Parameters
    -----------

    alpha: alpha value for Ridge, Lasso, ElasticNet  
            Default: 0.1  
        
    l1_ratio: Lasso Ratio for Better ElasticNet Gradient and MSE  
        Default: 0.5  

    Returns
    -------
    None

    Example
    -------

    >>> from rslearn.linear_model import ElasticNet
    >>> En = ElasticNet() # Using Default Parameters
    >>> En.fit(X, y, Scale=True) # Auto Scales Basicly for better performence use StandardScaler
    >>> En.predict(X_new)

    """


    def __init__(self, alpha=0.1, l1_score=0.5):
        self.alpha = alpha
        self.l1_score = l1_score
        self.weights, self.bias = None
        self.model = None
        self.type = "regression"
    
    def fit(self, X, y, scale=True, min_loss : float = 0.1):
        """
        `fit()` Function For `ElasticNet` to Train The Model  

        Parameters
        ----------

        X: array-like, 1D or 2D metrics for train model  

        y: array-like, 1D or 2D metrics of Real Value for X features

        Scaler: option to Scale Automaticly
            Default: True, Use `StandardScaler` and select `False` For better results
        
        Returns
        -------
        None
        """


        model = LinearRegression(regulization="elastic_net", alpha=self.alpha, l1_ratio=self.l1_score)
        
        model.fit(X, y, scale=scale, verbose=False, min_loss=min_loss)

        self.weights, self.bias = model.get_weight_bias()
        self.model = model
    
    def predict(self, X_new):
        prediction = self.model.predict(X_new)
        return prediction
    
    def get_weight_bias(self):
        return (self.weights, self.bias)
    
    def evaluate(self, X=None, y_pred=None, y_true=None):
        return self.model.evaluate(X=X, y_pred=y_pred, y_true=y_true)
