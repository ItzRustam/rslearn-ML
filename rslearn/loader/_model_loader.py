import gzip
import json
from rslearn.linear_model import LinearRegression, LogisticRegression
from rslearn.Errors import *
import numpy as np

def model_selector(model_name, weights = None, bias = None):
    if weights is None or bias is None:
        raise InternelError("Model File Curropted, report Issue on Github")
    weights = np.asarray(weights)
    model_list = {   
        'LinearRegression' : LinearRegression(weights=weights, bias=bias),
        'LogisticRegression' : LogisticRegression(),
    }

    return model_list[model_name]

def load_model(file_path : str = ""):
    if len(file_path) == 0:
        raise Error(f"Invalid file_path, {file_path}")
    
    with open(file_path, "rb") as f:
        compressed = f.read()

    json_bytes = gzip.decompress(compressed)

    model_data = json.loads(json_bytes.decode("utf-8"))


    model = model_selector(model_name=model_data['model'], weights=model_data['weights'], bias=model_data['bias'])
    model._fitted = True
    model.fitted_shape = np.asarray(model_data['fitted_shape'])

    if model_data['primary scaled']:
        model.Scaler.mean = np.asarray(model_data['scaler']['true']['mean'])
        model.Scaler.std = np.asarray(model_data['scaler']['true']['std'])
        model.Scaler._fitted = True
        model.flag = True
        # Scaler is Ready
    else:
        model.backup_scaler.maxx = model_data['scaler']['false']['max']
        model.backup_scaler._fitted = True
    
    return model