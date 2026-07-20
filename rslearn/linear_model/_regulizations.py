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

# from rslearn.linear_model import LinearRegression
from rslearn.BaseEstimators import BaseEstimatorRegulization

"""
This File Contains regulizing algorithams to avoid overfitting though the model  

Algorithams Contains  
----------------------
- `Lasso`
- `Ridge`
- `ElasticNet`
"""

class Lasso(BaseEstimatorRegulization):

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

    def __init__(self, alpha=0.1, l1_ratio=0.5, min_loss=0.1, max_itr=3000, hard_scale_off=False):
        super().__init__(alpha=alpha, l1_ratio=l1_ratio, regulization="l1", min_loss=min_loss, max_itr=max_itr, hard_scale_off=hard_scale_off)
    
    def fit(self, X, y, scale=True,):

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

        super().fit(X=X, y=y, scale=scale)


class Ridge(BaseEstimatorRegulization):
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
    

    def __init__(self, alpha=0.1, l1_ratio=0.5, min_loss=0.1, max_itr=3000, hard_scale_off=False):
        super().__init__(alpha=alpha, l1_ratio=l1_ratio, regulization="l2", min_loss=min_loss, max_itr=max_itr, hard_scale_off=hard_scale_off)
    
    def fit(self, X, y, scale=True,):

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

        super().fit(X=X, y=y, scale=scale)

class ElasticNet(BaseEstimatorRegulization):
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


    def __init__(self, alpha=0.1, l1_ratio=0.5, min_loss=0.1, max_itr=3000, hard_scale_off=False):
        super().__init__(alpha=alpha, l1_ratio=l1_ratio, regulization="elastic_net", min_loss=min_loss, max_itr=max_itr, hard_scale_off=hard_scale_off)
    
    def fit(self, X, y, scale=True,):

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

        super().fit(X=X, y=y, scale=scale)
    
