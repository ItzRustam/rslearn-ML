# User Guide
Complete Guide for users of Library, Contains every Information about Algorithms & Classes.  

**Make Sure you have already setup your environment.**  

---

## Architecture Summary

| Module | Purpose | Key Classes & Functions |
|---|---|---|
| [`linear_model`](linear_model/README.md) | Gradient-descent based linear models | `LinearRegression`, `LogisticRegression`, `Ridge`, `Lasso`, `ElasticNet` |
| [`neighbors`](neighbors/README.md) | K-Nearest Neighbors instance learning | `KNNClassifier`, `KNNRegressor` |
| [`Pipeline`](pipeline/README.md) | End-to-end automated ML pipeline | `pipeline` |
| [`preprocessing`](preprocessing/README.md) | Feature standardization & normalization | `StandardScaler`, `MinMaxScaler` |
| [`model_selection`](model_selection/README.md) | Dataset partitioning & CV | `train_test_split`, `GridSearchCV` |
| [`metrics`](metrics/README.md) | Model evaluation & loss functions | `r2_score`, `mse`, `mae`, `rmse`, `accuracy_score`, `confusion_metrics`, `precision`, `recall`, `f1_score`, `evaluate_model`, `EuclidienDisctance` |
| [`loader`](loader/README.md) | Pre-trained model loading | `load_model` (`.rslr`, `.rslc`) |
| [`Errors`](errors/README.md) | Library exception hierarchy | `Error`, `LengthError`, `NotFittedError`, `InvalidShape`, `InvalidValueError`, `InternelError` |
| [`BaseEstimators`](base_estimators/README.md) | Foundational OOP architecture & mixins | `BaseEstimator`, `BaseEstimatorKNN`, `BaseEstimatorRegulization`, `backupScaler` |

---

## Table Of Content

* <a href="linear_model/README.md">linear_models</a>  
   * <a href="linear_model/linear_regression.md">LinearRegression</a>  
   * <a href="linear_model/logistic_regression.md">LogisticRegression</a>  
   * <a href="linear_model/regulizations.md">regulizations (Lasso, Ridge, ElasticNet)</a>  

* <a href="neighbors/README.md">neighbors (KNN Family)</a> 
   * <a href="neighbors/KNNRegressor.md">KNN Regressor</a> 
   * <a href="neighbors/KNNClassifier.md">KNN Classifier</a> 

* <a href="pipeline/README.md">Pipeline (Automated Workflow)</a>
   * <a href="pipeline/pipeline.md">pipeline</a>

* <a href="preprocessing/README.md">preprocessing (scalers)</a> 
   * <a href="preprocessing/standard_scaler.md">StandardScaler (main Scaler)</a> 
   * <a href="preprocessing/minmax_scaler.md">MinMaxScaler</a> 

* <a href="model_selection/README.md">Model Selection (splitting & CV)</a> 
   * <a href="model_selection/splitting.md">train_test_split</a> 
   * <a href="model_selection/grid_search_cv.md">GridSearchCV (Beta)</a> 

* <a href="metrics/README.md">Metrics (evaluations & distances)</a> 
   * <a href="metrics/classification.md">Classification Metrics (accuracy, precision, recall, f1, confusion matrix)</a> 
   * <a href="metrics/regression.md">Regression Metrics (r2, mse, mae, rmse)</a> 
   * <a href="metrics/evaluate_model.md">evaluate_model (All-in-One Evaluator)</a>
   * <a href="metrics/distances.md">Distance Metrics (EuclidienDisctance)</a>

* <a href="loader/README.md">Saved Model Loader</a> 
   * <a href="loader/load_model.md">Load Model (.rslr, .rslc)</a> 

* <a href="errors/README.md">Errors (Exception Hierarchy)</a>
   * <a href="errors/README.md">Custom Exception Reference</a>

* <a href="base_estimators/README.md">BaseEstimators (OOP Architecture)</a>
   * <a href="base_estimators/README.md">Base Classes, backupScaler & Developer Guide</a>
