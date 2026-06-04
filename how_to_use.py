from rslearn.linear_model import LinearRegression # Importing Regression
from rslearn.preprocessing import MinMaxScaler # Importing Scaler
import numpy as np


X = np.array([10 , 20, 30, 40, 50, 60])
y = np.array([100, 200, 300 ,400 ,500, 600])
X_test = np.array([60, 70, 80])

# Without Pipeline

Model = LinearRegression()
Model.fit(X, y,)
Evaluations = Model.evaluate(X=X_test, y_true=np.array([600, 700, 800]))
print(Evaluations)

# It will automaticly Predict and print Evaluations like r2_score, mse, rmse, mae
# and automaticly Scale

"""
OUTPUT
=======
{
    'prediction': array([600., 699., 799.]), 
    'evaluation': {
            'r2_score': np.float64(0.9999),
            `mse': 0.6666666666666666,
            'mae': 0.6666666666666666,
            'rmse': 0.816496580927726
    }
}

SAME FOR LOGISTIC REGRESSION

"""