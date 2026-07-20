import gzip
import json
from rslearn.linear_model import LinearRegression, LogisticRegression
from rslearn.Errors import *
import numpy as np
    

def load_logistic(file_path : str = "rslearn_model.rslc"):
    with open(file_path, "rb") as f:
        compressed = f.read()

    json_bytes = gzip.decompress(compressed)

    model_data = json.loads(json_bytes.decode("utf-8"))
    

    if "rslearn_compressed" not in model_data:
        raise Error("Invalid compressed file. giving file is not compressed by rslearn.")
    
    if model_data["task"] != "classification":
        raise Error("Invalid classification Model.")
    
    if model_data['solver'] == 'liblinear':
        weights = np.array(model_data["solver_options"]["liblinear"]['weights'])
        model : LogisticRegression = LogisticRegression(solver=model_data['solver'], weights=weights, bias=model_data["solver_option"]["liblinear"]["weights"], max_itr=model_data["params"]["max_itr"], lr=model_data["params"]["lr"], hard_scale_off=model_data["hard_scale_off"])
    elif model_data['solver'] == 'saga':
        catog_models_ = []
        models_info : dict = model_data['solver_options']['saga']['catogirical_models']
        for key in models_info.keys():
            weights = np.array(models_info[key]['weights'])
            mod = LogisticRegression(solver='liblinear', weights=weights, bias=models_info[key]['bias'])
            catog_models_.append(mod)
        
        model : LogisticRegression =  LogisticRegression(solver='saga', catogirical_model=catog_models_, max_itr=model_data["params"]["max_itr"], lr=model_data["params"]["lr"], hard_scale_off=model_data["hard_scale_off"])

    else:
        raise InternelError("File is curropted. Invalid solver.")


    # model : LogisticRegression = LogisticRegression()
    model._fitted = True
    model.fitted_shape = np.asarray(model_data['fitted_shape'])
    if model_data['hard_scale_off']:
        return model
    # Else cases
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


def load_linear(file_path : str = "rslearn_model.rslr"):
    
    with open(file_path, "rb") as f:
        compressed = f.read()

    json_bytes = gzip.decompress(compressed)

    model_data = json.loads(json_bytes.decode("utf-8"))

    if "rslearn_compressed" not in model_data:
        raise Error("Invalid compressed file. giving file is not compressed by rslearn.")

    if model_data["task"] != "regression":
        raise Error("Invalid Regression Model.")

    # model : LinearRegression = model_selector(model_name=model_data['model'], weights=model_data['weights'], bias=model_data['bias'])
    model = LinearRegression(regulization=model_data["params"]["regulization"], alpha=model_data["params"]["alpha"], l1_ratio=model_data["params"]["l1_ratio"], min_loss=model_data["params"]["min_loss"], lr=model_data["params"]["lr"], weights=np.array(model_data["weights"]), bias=model_data["bias"], hard_scale_off=model_data["hard_scale_off"], max_itr=model_data["params"]["max_itr"])


    model._fitted = True
    model.fitted_shape = np.asarray(model_data['fitted_shape'])
    # model.hard_scale_off = model_data['hard_scale_off']
    if model_data['hard_scale_off']:
        return model
    # Else cases
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

def load_model(file_path : str = "rslearn_model.rsl"):
    if len(file_path) == 0:
        raise Error(f"Invalid file_path, {file_path}")
    
    valid_extensions = ['.rslr', 'rslc']
    if not(file_path.lower().endswith(tuple(valid_extensions))):
        raise Error(f"Invalid Extension {file_path}, supported {valid_extensions}")
    
    if file_path.endswith(".rslr"):
        model = load_linear(file_path=file_path)
        return model
    
    if file_path.endswith(".rslc"):
        model = load_logistic(file_path=file_path)
        return model