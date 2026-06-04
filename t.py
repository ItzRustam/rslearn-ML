from rslearn.linear_model import LogisticRegression
from rslearn.Pipeline import pipeline
from rslearn.preprocessing import StandardScaler
import numpy as np

X = np.array([[20, 12], [13, 17], [12, 10], [9, 7]])
y = np.array([1, 0, 1, 0])


# Model = LogisticRegression()
# Model.fit(X, y)

t = np.array([[12, 8], [7, 5]])

# ev = Model.evaluate(X=t, y_true=np.array([1, 0]))
# print(ev)

# print("DONEEEE!!!")

# evn = Model.evaluate(y_pred=np.array([1, 2, 0]), y_true=np.array([1, 2, 0]))
# print(evn)

# ano = LinearRegression()
# ano._fitted = True

# ennv = ano.evaluate(y_true=np.array([12.3, 45.2, 12.1]), y_pred=np.array([12.2, 45.1, 11.9]))
# print(ennv)




pipe = pipeline(params={'model': LogisticRegression(), 'scaler': StandardScaler()})
pipe.fit(X, y)

ev = pipe.evaluate(X=t, y_true=np.array([1, 0]))
print(ev)