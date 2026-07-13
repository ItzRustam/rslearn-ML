import numpy as np
from rslearn.preprocessing import StandardScaler
from rslearn.BaseEstimators import backupScaler # X/(max(X) + 1e-9)
from rslearn.BaseEstimators import _base
from abc import ABC, abstractmethod
from rslearn.Errors import *
from rslearn.metrics import *



# Task - Complete This Class.
class BaseEstimator(ABC):
    def __init__(self, lr : float = 0.001, max_itr : int =1000, weights : np.array = None, bias : float = None):
        self.lr = lr
        self.max_itr = max_itr
        self.weights = weights
        self.bias = bias
        self.Scaler = StandardScaler() # Scaler
        self.backup_scaler = backupScaler()
        self.flag = False # Flag For Scaler Status
        self.type = "base"
        self.fitted_shape = None # Edge Case
        self._fitted = False # Edge Case
    
    # If scale=True
    def _scale_True(self, X, scaled=False):
        if scaled:
            return self.Scaler.transform(X)
        else:
            return self.Scaler.fit_transform(X)
    
    # If scale=False
    def _scale_False(self, X, scaled=False):
        if scaled:
            return self.backup_scaler.transform(X)
        else:
            return self.backup_scaler.fit_transform(X)

    
    def fit(self, X, y, scale=True):
        pass 

    def predict(self, X):
        pass

    def evaluate(self, X=None, y_pred=None, y_true=None):
        pass

    # Helper if self.type=regression
    def _eval_help_regression(self, X=None, y_pred=None, y_true=None):
        if not self._fitted: # If Model is not fitted
            raise NotFittedError(
                "This model is not trained yet. Call 'fit()' before using 'evaluate()'."
            )

        if y_true is None: # Edge case : Nothing to compare
            raise InvalidValueError("Invalid Arguments `y_true` `None`")
        
        
        if y_pred is None:
            if X is None: # Edge case: Both `X` and `y_pred` are None
                raise InvalidValueError("parameter `X` and `y_pred` Both given None.")
        
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

    # Helper if self.type=classification
    def _eval_help_classification(self, X=None, y_pred=None, y_true=None):
        if not self._fitted: # If Model is not fitted
            raise NotFittedError(
                "This model is not trained yet. Call 'fit()' before using 'evaluate()'."
            )

        if y_true is None: # Edge case : Nothing to compare
            raise InvalidValueError("Invalid Arguments `y_true` `None`")
        
        
        if y_pred is None:
            if X is None: # Edge case: Both `X` and `y_pred` are None
                raise InvalidValueError("Both `X` and `y_pred` are None.")
        
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
    

    def _eval(self, X=None, y_pred=None, y_true=None):
        if self.type == "base":
            raise InternelError("Class is Not Registerd as `classification` or `regression`")
        elif self.type == "regression":
            return self._eval_help_regression(X=X, y_pred=y_pred, y_true=y_true)
        elif self.type == "classification":
            return self._eval_help_classification(X=X, y_pred=y_pred, y_true=y_true)
        else:
            raise Error(f"Invalid Class type `{self.type}`")
        




    