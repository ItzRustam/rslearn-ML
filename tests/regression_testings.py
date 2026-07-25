import numpy as np
from rslearn.neighbors import KNNRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from rslearn.model_selection import train_test_split
from pprint import pprint

# d = fetch_california_housing()

# X = d.data
# y = d.target

# print(len(y))

# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=67, test_size=0.25)

X_train = [10, 20, 30]
y_train = [34, 64, 94]
X_test = [20, 40, 10]
y_test = [64, 124, 34]

# rsm = KNNRegressor(k_neighbors=12 ,hard_scale_off=False) # 68 r2_score.
# rsm.fit(X_train, y_train, scale=True)
# pred = rsm.predict(X_test)
# pprint(rsm.evaluate(y_pred=pred, y_true=y_test))

# data = np.array([
#         [1,2],
#         [3, 4],
#         [4, 5]
#     ])

# key = np.array([0, 1, 1])
# model = KNNClassifier(k_neighbors=2)
# model.fit(data, key)

# new = np.array([
#     [2, 3],
#     [0, 1]
# ])

# pred = model.predict(new)
# print(pred)

# m = KNeighborsClassifier(n_neighbors=2)
# m.fit(data, key)

# pred = model.predict(new)
# print(pred)

# from rslearn.BaseEstimators import BaseEstimator

# bs = BaseEstimator()
# bs.type = "regression"
# # bs._eval()
# print(type(bs))
# print(bs.type)

from rslearn.linear_model import LinearRegression # Importing Regression
# from rslearn.preprocessing import MinMaxScaler # Importing Scaler
import numpy as np
from rslearn.loader import load_model



# Without Pipeline
# weight = np.array([170.73251277])

# Model = LinearRegression(max_itr=18000, regulization="l1")
# Model.fit(X_train, y_train, scale=False)
# Evaluations = Model.evaluate(X=X_test, y_true=y_test)
# print(Evaluations)
# print(Model.get_weight_bias())
# print(Model.regulization, Model.max_itr)
# print(Model.save("first_model.rsl"))
# model = load_model("first_model.rslr")
# Evaluations = model.evaluate(X=X_test, y_true=y_test)
# print(Evaluations)
# print(model.get_weight_bias())
# print(model.max_itr)
# print(model.regulization)

# m = LinearRegression(regulization=None ,max_itr=3000, hard_scale_off=False)
# m.fit(X, y, scale=True) # Or Even False both will be ignored!
# print(m.evaluate(X=X_test, y_true=[600, 700, 800]))
# # print(m.save("hard_scale_offf.rsl"))

# modl = load_model("hard_scale_offf.rslr")
# print(modl.evaluate(X=X_test, y_true=[600, 700, 800]))

# print(modl._model)

# model.fit(X_train, y_train, scale=False)
# Evaluations = model.evaluate(X=X_test, y_true=y_test)
# print(Evaluations)
# print(model.flag)
# print(model.get_weight_bias())

X = np.array([10 , 20, 30, 40, 50, 60])
y = np.array([100, 200, 300 ,400 ,500, 600])
X_test = np.array([60, 70, 80])

# Train on Diffrent Data
# print(model.fitted_shape)
# model.fit(X, y, scale=True)
# Evaluations = model.evaluate(X=X_test, y_true=[600, 700, 800])
# print(Evaluations)
# print(model.flag)
# print(model.get_weight_bias())
# print(model.fitted_shape)

from rslearn.linear_model import Lasso
from rslearn.Pipeline import pipeline
from rslearn.preprocessing import MinMaxScaler, StandardScaler
from rslearn.loader import load_pipeline
# LassoM = Lasso()
# LassoM.fit(X=X, y=y)
# print(LassoM.evaluate(X=X_test, y_true=[600, 700, 800]))
# LassoM.save("lasso_model.rsl")

# m = load_model("lasso_model.rslr")
# print(m.evaluate(X=X_test, y_true=[600, 700, 800]))

line = pipeline({'model': Lasso(), 'scaler': StandardScaler()}, validation_split=False, split_params={"test_size":0.5, "random_state":67, "stratify":None})
line.fit(X, y)
print(line.evaluate(X=X_test, y_true=[600, 700, 800]))
print(line.Model.get_weight_bias())
print(line.save("first_line.prsl"))

lineOp = load_pipeline("first_line.prsl" ,"pipeline_Lasso.rslr")
print(lineOp.evaluate(X=X_test, y_true=[600, 700, 800]))
print(lineOp.scaling)
print(lineOp.Model.get_weight_bias())
print(lineOp.predict(X_test))

print(lineOp.evaluate(y_true=[600, 700, 800], y_pred=line.predict(X_test)))
print(lineOp.Model.hard_scale_off)

lineOp.fit(X, y)
print(lineOp.evaluate(X=X_test, y_true=[600, 700, 800]))
print(lineOp.Model.get_weight_bias())
print(lineOp.predict(X_test))