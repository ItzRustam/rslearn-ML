import numpy as np
from rslearn.preprocessing import StandardScaler
from rslearn.BaseEstimators import _base
from abc import ABC, abstractmethod
from rslearn.Errors import *
from rslearn.metrics import *
import rslearn
import json
import gzip
from typing import List

"""Secodary Scaler"""
class backupScaler:
    def __init__(self, epsilon=1e-9):
        self.maxx = 0
        self.epsilon = epsilon
        self._fitted = False
    
    def fit(self, X):
        self.maxx = np.max(X)
        self._fitted = True
    
    def transform(self, X):
        if self._fitted:
            new_X = X/(self.maxx + self.epsilon) # avoid devisible by 0
            return new_X
        else:
            raise NotFittedError("Scaler has not been fitted yet.")
    
    def fit_transform(self, X):
        self.fit(X=X) # Fitting
        scaled = self.transform(X=X)
        return scaled

class BaseEstimator(ABC):
    def __init__(self, lr : float = 0.001, max_itr : int =3000, weights : np.array = None, bias : float = None):
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
        self._model = "BaseEstimator"
        self._lib_version = rslearn.__version__
        self.hard_scale_off = None
    
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
            "r2_score": float(r2_Score),
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
    
    def get_weight_bias(self) -> tuple:
        """Input = None, 
        O/P - (np.array, float64)
        >>> weights, bias = Model.get_weight_bias()
        """
        return (self.weights, self.bias)
    
    def __save_regressor_model(self, file_name : str = "rslearn_model.rslr", regulization=None, min_loss=0.1, alpha=0.1, l1_ratio=0.5):
        model_data = {
            "model": self._model,
            "version": self._lib_version,
            "rslearn_compressed": True,
            "task" : "regression",
            "primary scaled": self.flag,
            "weights": self.weights.tolist(),
            "bias": float(self.bias),
            "hard_scale_off" : self.hard_scale_off,
            "fitted_shape" : self.fitted_shape,
            "scaler": {
                "true": {
                    "scaler_name": "StandardScaler",
                    "mean": self.Scaler.mean.tolist(),
                    "std": self.Scaler.std.tolist()
                },
                "false": {
                    "scaler_name": "BackupScaler",
                    "max": float(self.backup_scaler.maxx)
                }
            },
            # For Retrainable format
            "params":{
                "regulization" : regulization,
                "min_loss" : min_loss,
                "alpha" : float(alpha[0]),
                "l1_ratio":l1_ratio,
                "lr" : self.lr,
                "max_itr" : self.max_itr
            }
        }

        json_bytes = json.dumps(model_data).encode("utf-8")

        compressed = gzip.compress(json_bytes)

        with open(file_name, "wb") as f:
            f.write(compressed)

        return f"Model Saved Successfully with {file_name}"
    

    def __save_classification_model(self, file_name : str = "rslearn_model.rslc", solver="liblinear", catogirical_models=[]):
        catog_models_ = {}

        if len(catogirical_models) == 0:
            pass
        else:
            for count, model in enumerate(catogirical_models):
                weight_bias = {
                    'weights' : model.weights.tolist(),
                    'bias' : float(model.bias)
                }
                catog_models_[f'model_{count}'] = weight_bias
                
        model_data = {
            "model": self._model,
            "version": self._lib_version,
            "rslearn_compressed": True,
            "task" : "classification",
            "primary scaled": self.flag,
            "solver" : solver,
            "solver_options":{
                "liblinear":{
                    "weights": self.weights.tolist(),
                    "bias": float(self.bias),
                },
                "saga":{
                    "catogirical_models" : catog_models_
                }
            },

            "hard_scale_off" : self.hard_scale_off,
            "fitted_shape" : self.fitted_shape,
            "scaler": {
                "true": {
                    "scaler_name": "StandardScaler",
                    "mean": self.Scaler.mean.tolist(),
                    "std": self.Scaler.std.tolist()
                },
                "false": {
                    "scaler_name": "BackupScaler",
                    "max": float(self.backup_scaler.maxx)
                }
            },

            "params":{
                "lr" : self.lr,
                "max_itr" : self.max_itr,

            }
        }

        json_bytes = json.dumps(model_data).encode("utf-8")

        compressed = gzip.compress(json_bytes)

        with open(file_name, "wb") as f:
            f.write(compressed)

        return f"Model Saved Successfully with {file_name}"
    

    def save(self, file_name : str = "rslearn_model.rsl", solver="liblinear", catogirical_models=[], regulization=None, min_loss=0.1, alpha=0.1, l1_ratio=0.5):
        if not(self._fitted):
            raise NotFittedError("Model has not been fitted yet, use `fit()`")
        if not(file_name.endswith(".rsl")):
            raise Error("Only `.rsl` format are supported")
        
        if self._model == "LinearRegression":
            self.__save_regressor_model(file_name=f"{file_name}r", regulization=regulization, min_loss=min_loss, alpha=alpha, l1_ratio=l1_ratio)
        elif self._model == "LogisticRegression":
            self.__save_classification_model(file_name=f"{file_name}c", solver=solver, catogirical_models=catogirical_models)
        else:
            raise Error(f"{self._model} does not support saving.")
            
        
class BaseEstimatorKNN(ABC):
    def __init__(self, k_neighbors : int = 5):

        self.k = k_neighbors
        self.Scaler = StandardScaler() # Scaler
        self.backup_scaler = backupScaler()
        self.flag = False # Flag For Scaler Status
        self.type = "base"
        self.fitted_shape = None # Edge Case
        self._fitted = False # Edge Case
        self._model = "BaseEstimator"
        self._lib_version = rslearn.__version__
        self.hard_scale_off = None
        # For Store The Data
        self.fitted_x = None
        self.fitted_y = None 

        if k_neighbors <= 0:
            raise InvalidValueError("k_neighbors can't be smaller than 1")
    
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
                "This model is not fitted yet. Call 'fit()' before using 'evaluate()'."
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
            "r2_score": float(r2_Score),
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
        



if __name__ == '__main__':
    bs = BaseEstimator()
    bs._eval()
