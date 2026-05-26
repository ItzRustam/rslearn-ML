from rslearn.linear_model import LinearRegression # Importing Regression
from rslearn.preprocessing import MinMaxScaler # Importing Scaler
import numpy as np


X = np.array([10 , 20, 30, 40, 50, 60])
y = np.array([100, 200, 300 ,400 ,500, 600])
X_test = np.array([60, 70, 80])

# Without Scaler - Not Prefered Or Change Learning_rate to 0.0001 or lower value

Model = LinearRegression()

Model.fit(X, y)
print(Model.get_weight_bias())
print(Model.predict(X_test))


# ___________________________________________
# With Scaler - Prefered

Scaler = MinMaxScaler() # you can also use StandardScaler
X = Scaler.fit_transform(X.reshape(-1, 1))
X_test = Scaler.transform(X_test.reshape(-1, 1))

Model.fit(X.reshape(-1), y.reshape(-1))
print(Model.get_weight_bias())
print(Model.predict(X_test))