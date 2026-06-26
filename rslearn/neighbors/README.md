# KNN Models Documentation

This document provides an overview of the K-Nearest Neighbors (KNN) implementation for both classification and regression tasks.

## Models Overview

### KNN Classifier

#### Description:
A classification model that implements the K-Nearest Neighbors algorithm for supervised learning. It supports optional data scaling during the fit phase to improve performance.

#### Parameters:
- `k_neighbors`: int, default=5 - Number of neighbors to consider.
- `Scaler`: StandardScaler, default=None - Scaler object used for feature scaling (must be initialized if `scale=True`).
- `scale`: bool, default=True - Whether to perform scaling on the input features before fitting.

#### Key Methods:
1. **fit(X, y, scale=True)**: 
   - Fits the model using the provided training data.
   - Can optionally enable scaling of features.
   
2. **predict(X_new)**:
   - Makes predictions for new input data based on fitted model.
   
3. **evaluate(X=None, y_pred=None, y_true=None)**:
   - Evaluates model performance by comparing predicted values with true labels.
   - Returns a dictionary containing both predictions and evaluation metrics.

#### Evaluation Metrics:
- `accuracy_score`: Accuracy of the predictions.
- `f1_score`: F1 score for classification tasks.
- `recall`: Recall score.
- `precision`: Precision score.

### KNN Regressor

#### Description:
A regression model that implements the K-Nearest Neighbors algorithm for supervised learning. It supports optional data scaling during the fit phase to improve performance.

#### Parameters:
- `k_neighbors`: int, default=5 - Number of neighbors to consider.
- `Scaler`: StandardScaler, default=None - Scaler object used for feature scaling (must be initialized if `scale=True`).
- `scale`: bool, default=True - Whether to perform scaling on the input features before fitting.

#### Key Methods:
1. **fit(X, y, scale=True)**: 
   - Fits the model using the provided training data.
   
2. **predict(X_new)**:
   - Makes predictions for new input data based on fitted model.
   
3. **evaluate(X=None, y_pred=None, y_true=None)**:
   - Evaluates model performance by comparing predicted values with true targets.
   - Returns a dictionary containing both predictions and evaluation metrics.

#### Evaluation Metrics:
- `r2_score`: Coefficient of determination (R²).
- `mse`: Mean Squared Error.
- `mae`: Mean Absolute Error.
- `rmse`: Root Mean Squared Error.

## Usage Instructions

### Basic Workflow:

1. **Import Required Components**:
   ```python
   from rslearn.neighbors import KNNClassifier, KNNRegressor
   ```

2. **Initialize Model**:
   ```python
   model = KNNClassifier(n_neighbors=5)  # For classification
   model_regressor = KNNRegressor(n_neighbors=5)  # For regression
   ```

3. **Prepare Data**:
   ```python
   # Example for classification
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
   
   # Example for regression
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
   ```

4. **Fit the Model**:
   ```python
   model.fit(X_train, y_train)
   model_regressor.fit(X_train, y_train)
   ```

5. **Make Predictions**:
   ```python
   predictions = model.predict(X_test)
   predictions_regressor = model_regressor.predict(X_test)
   ```

6. **Evaluate Model**:
   ```python
   evaluation = model.evaluate(X_test, y_test)  # For classification
   evaluation_regressor = model_regressor.evaluate(X_test, y_test)  # For regression
   ```

### Important Notes:

- Handle the scaling parameter appropriately based on your data.
- Ensure that during prediction phase, if no new data is provided, predictions are generated using the fitted model's stored training data.
- The KNN algorithm uses Euclidean distance to find nearest neighbors.

## Technical Details

- **Distance Calculation**: Uses EuclidienDisctance as the distance metric for measuring similarity between samples.
- **Data Scaling**: By default, features are scaled using StandardScaler. This helps in improving model performance by normalizing feature values.
- **K Value Selection**: Choose an appropriate number of neighbors (k) based on your specific dataset and problem.

## Implementation Steps

1. **Import Necessary Libraries**:
   ```python
   from rslearn.neighbors import KNNClassifier, KNNRegressor
   from rslearn.preprocessing import StandardScaler
   ```

2. **Initialize Model Components**:
   ```python
   scaler = StandardScaler()
   model = KNNClassifier(scale=True)  # Initialize classifier with scaling enabled
   ```

3. **Fit the Model**:
   ```python
   model.fit(training_data, target_labels)
   ```

4. **Predict Using New Data**:
   ```python
   new_data = ...  # Your test/prediction data
   predictions = model.predict(new_data)
   ```

5. **Evaluate Model Performance**:
   ```python
   results = model.evaluate(test_data, true_labels)  # Returns both predictions and metrics
   ```

## Example Usage

### For Classification:

```python
from rslearn.neighbors import KNNClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load Iris dataset
data = load_iris()
X = data.data
y = data.target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Initialize KNN classifier with k=3
model = KNNClassifier(n_neighbors=3)

# Fit the model
model.fit(X_train, y_train)

# Make predictions on test data
predictions = model.predict(X_test)

# Evaluate the model
evaluation = model.evaluate(X_test, y_test)
print("Model predictions:", predictions)
print("Evaluation metrics:", evaluation)
```

### For Regression
```python
from rslearn.neighbors import KNNRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

# Load Boston housing dataset
data = load_boston()
X = data.data
y = data.target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Initialize KNN regressor with k=3
model = KNNRegressor(n_neighbors=3)

# Fit the model
model.fit(X_train, y_train)

# Make predictions on test data
predictions = model.predict(X_test)

# Evaluate the model
evaluation = model.evaluate(X_test, y_test)
print("Model predictions:", predictions)
print("Evaluation metrics:", evaluation)
```