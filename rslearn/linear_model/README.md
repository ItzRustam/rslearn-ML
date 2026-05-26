# 📊 linear_models 
Collection of most of commonly used linear_model like Linear Regression Logistic Regression
Scaler is preferd for better result

## 🤖 contains Models
- Linear Regression
- Logistic Regression
- Ridge & Lasso & ElasticNet Regulization (l1, l2 and ElasticNet)

## 🆕 Latest Feature
- `evaluate()` method to Auto calculate All metrics E.g r2_score etc

**Parameters**  
| Parameter | Description |
|-----------|-------------|
| `X`| new data to predict from Model |  
| `y_pred` | prediction from Model Require when X = None |
| `y_true` | correct output for X |

`y_true` is mendantory  
Enter `X` or `y_pred` Anyone of them  

## How to use
its pretty Simple Just import define and fit and then predict like sklearn like  

### `Linear Regression`
**new: Added Evaluation()**

#### `regulization` options
| Option            | Description                             |
| ----------------- | --------------------------------------- |
| `l1`              | Lasso Regulization                      |
| `l2`              | Ridge Regulization                      |
| `ElasticNet`      | ElasticNet Regulization                 |

***Read Doc Strings For More Prameter Knowledge***

```python
from rslearn.linear_model import LinearRegression
Model = LinearRegression(regulization="l1")
```

#### 🆕 New Parameter - `scale` in fit()
_Uses StandardScaler to Scale X for Train and Predict Both Time Automatically if its True
`Default: True`_   

#### 🆕 New Method - `evaluate()`
Calculates All metrics/* like r2_score, mse, rmse, mae etc.  

When `y_pred` not given
``` bash
metrics_output = Model.evaluate(X=X, y_true=y_true)
```
Output -  
``` python
{  
    "prediction": [...],   
    "evaluate": {
        "r2_score": score,
        "mse": mse,
        "mae": mae,
        "rmse": rmse
    }

}
```

When `y_pred` is given
``` bash
metrics_output = Model.evaluate(y_pred=y_pred, y_true=y_true)
```
Output -  
``` python
{  
    "prediction": [y_pred],   
    "evaluate": {
        "r2_score": score,
        "mse": mse,
        "mae": mae,
        "rmse": rmse
    }

}
```

### `Logistic Regression`
StandardScaler or MinMaxScaler is preferd in Multi class classification
``` python
from rslearn.linear_model import LogisticRegression
Model = LogisticRegression()
```

#### 🆕 New Parameter - `scale` in fit()
_Uses StandardScaler to Scale X for Train and Predict Both Time Automatically if its True
`Default: True`_  

checkout preprocessing/README.md for Scalers detail

#### 🆕 New Method - `evaluate()`
Calculates All metrics/* like accuracy_score, recall, f1_score, precision etc.  

When `y_pred` not given
``` bash
metrics_output = Model.evaluate(X=X, y_true=y_true)
```
Output -  
``` python
{  
    "prediction": [...],   
    "evaluate": {
        "accuracy_score": score,
        "recall": recall,
        "precision": precision,
        "f1_score": F1
    }

}
```

When `y_pred` is given
``` bash
metrics_output = Model.evaluate(y_pred=y_pred, y_true=y_true)
```

Output -  
``` python
{  
    "prediction": [y_pred],   
    "evaluate": {
        "accuracy_score": score,
        "recall": recall,
        "precision": precision,
        "f1_score": F1
    }

}
```


Thats It! 

### `Ridge`, `Lasso`, `ElasticNet`
Regulizations For avoid overfitting

`New parameter`:  
`Scale=True` Automaticly Scale Data before sending to LinearRegression,  
Use `Scalers`, e.g `StandardScaler`, `MinMaxScaler` for better performance

``` python
from rslearn.linear_model import Lasso, Ridge, ElasticNet
```

#### 🆕 New Method - `evaluate()`
Calculates All metrics/* like r2_score, mse, rmse, mae etc. 

**Same As Linear or Logistic Regression**


### Documentation is coming! Explained All Parameters In that.
### `More Coming Soon`

#### Maden with ♥