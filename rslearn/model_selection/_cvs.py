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
_cvs.py by Rustam Singh Bhadouoriya

This File Contains Source for GridSearchCV and Future CVs.

Availble
--------
`GridSearchCV`
"""



import numpy as np
from rslearn.model_selection import train_test_split # For Auto Splitting
from rslearn.preprocessing import StandardScaler
from rslearn.Pipeline import pipeline # For Automate work (auto splitting mode)
from rslearn.metrics import (accuracy_score,
                            f1_score, 
                            recall, 
                            precision,
                            r2_score, 
                            mse,
                            rmse,
                            mae
                            ) # Complete Metrics needed in this Case


"""LOGIC
input may be some parameters like alpha and max_itr which are dict and came in this format

Model,
auto_split=False
{
"alpha": [1, 2, 3, 4, 5],
"max_itr": [1800, 100, 400, 800]
"Scaler": [StandardScaler(), MinMaxScaler()] # to check which perform better.
}

WHAT WE WILL DO :-
We will make diffrent diffrent pipeline with diffrent diffrent prameters
and check its evaluation which evaluation is better on that model then it will be stored.

"""

# NOTE: Beta Program

class GridSearchCV:

    def _check_valid_score(scoring, model):
        if scoring == None: # If Scoring is Not Given
            if model.type == "regression":
                return "r2_score"
            else:
                return "accuracy_score"

        valid_scoring_methords = ["r2_score", "accuracy_score", "mse", "rmse", "mae", "mean_squared_error", "root_mean_squared_error", "mean_absolute_error", "f1_score", "recall", "precision"]
        if scoring not in valid_scoring_methords:
            raise ValueError(f"Invalid Scoring Methord Only {valid_scoring_methords} exits")
        
        best_match_map = {
            "r2_score" : "r2_score",
            "accuracy_score": "accuracy_score",
            "mean_squared_error": "mse",
            "root_mean_squared_error": "rmse",
            "mean_absolute_error": "mae",
            "precision": "precision",
            "recall": "recall",
            "f1_score": "f1_score"
        }

        if scoring in best_match_map.values():
            return scoring
        
        return best_match_map[scoring] # If scoring is kinda indirect
        


    def __init__(self, model=None, scoring=None, params={}, cv=2):
        if len(params) == 0:
            raise ValueError("Empty Parameter For GridSearchCV")
        
        if model==None:
            raise ValueError("`model` is not provided")
        
        if scoring==None:
            print("No Manual Socoring Found - Using Default For Model")

        
        self.Model = model
        self.scaling = False

        self.params = params

        if "scaler" in params:
            self.Scaler = params["scaler"]
            self.scaling = True
        
        self.trained = False # Flag to Train
        self.scoring_methord = self._check_valid_score(scoring, model)
        # scoring = r2_score or accuracy_score if None
        self.cv = cv

    
    def fit(self, X=None, y=None):
        if X == None or y == None:
            raise ValueError("`None` value found in input.")
        X = np.asarray(X)
        y = np.asarray(y)
        
        self.best_model = None
        test_model = None

        scores = 0.0

        # NOTE: for Logistic Only max_itr, and Linear it is alpha, l1_ratio, max_itr
        for opr in range(len(self.params)):
            for x in opr:

                test_model = self.Model

                if test_model.type == "classification": # Logistic
                    test_model.fit(X, y, scale=self.scaling, max_itr=x)

                    # scoring
                    accuracy = accuracy_score(y, test_model.predict(X))
                    if accuracy >= scores:
                        scores = accuracy

                        # setting best model

                        self.best_model = test_model
                
                else:
                    test_model.fit(X, y,)

                

                    


        

        
    

    
        


    