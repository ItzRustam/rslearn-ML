import numpy as np
from rslearn.neighbors import KNNClassifier
# from sklearn.neighbors import KNeighborsClassifier
from rslearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from rslearn.model_selection import train_test_split
from rslearn.metrics import accuracy_score
from pprint import pprint
from rslearn.preprocessing import *

d = load_iris()

X = d.data
y = d.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=67, test_size=0.25, stratify=y)

from rslearn.Pipeline import pipeline
from rslearn.loader import load_pipeline

line = pipeline(params={'model': LogisticRegression(), 'scale' : MinMaxScaler()})
line.fit(X_train, y_train)
print(line.evaluate(X=X_test, y_true=y_test))
line.save("iris_pipeline.prsl")

lineS = load_pipeline("iris_pipeline.prsl", "pipeline_LogisticRegression.rslc")


# rsm = KNNClassifier(hard_scale_off=True)
# rsm.fit(X_train, y_train, scale=True)
# pred = rsm.predict(X_test)
# print(accuracy_score(y_test, pred))
# pprint(rsm.evaluate(y_pred=pred, y_true=y_test))

# mod = LogisticRegression(solver="saga")
# mod.fit(X_train, y_train)
# print(mod.flag)
# print(mod.evaluate(X=X_test, y_true=y_test))
# print(mod._cato_model)
# print(mod.get_weight_bias())
# print(mod._model)
# print(mod._lib_version)


# Model Saving :)
# modls = mod._cato_model

# modd = LogisticRegression(solver="saga", catogirical_model=modls)
# modd._fitted = True
# modd.fitted_shape = X_train.shape
# modd.flag = True
# modd.Scaler.mean = mod.Scaler.mean
# modd.Scaler.std = mod.Scaler.std
# modd.Scaler._fitted = True
# print(modd.evaluate(X=X_test, y_true=y_test))
# print(modd._cato_model)
# print(modd.get_weight_bias())
# print(modd._model)
# print(modd._lib_version)

# print(mod.save("iris.rsl"))

# from rslearn.loader import load_model
# model = load_model("iris.rslc")
# print(model.evaluate(X=X_test, y_true=y_test))
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
