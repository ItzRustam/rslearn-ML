import gzip
import json
# from rslearn.linear_model import LinearRegression, LogisticRegression
from rslearn.Pipeline import pipeline
from rslearn.Errors import *
from rslearn.loader import load_model
from rslearn.preprocessing import *
import numpy as np

def pipeline_load_helper(pipeline_data : dict, model_file : str = None) -> pipeline:
    # Model Ready
    model = load_model(model_file)
    if pipeline_data["scaling"]:
        if pipeline_data["scaler"] == "MinMaxScaler":
            scaler = MinMaxScaler()
            scaler.min_v = np.array(pipeline_data["MinMaxScaler"]["min"])
            scaler.max_v = np.array(pipeline_data["MinMaxScaler"]["max"])
            scaler.a, scaler.b = pipeline_data["MinMaxScaler"]["a"], pipeline_data["MinMaxScaler"]["b"]
            scaler._fitted = True

        elif pipeline_data["scaler"] == "StandardScaler":
            scaler = StandardScaler()
            scaler.mean = np.array(pipeline_data["StandardScaler"]["mean"])
            scaler.std = np.array(pipeline_data["StandardScaler"]["std"])
            scaler._fitted = True
        
        line = pipeline({'model' : model, 'scaler' : scaler}, validation_split=pipeline_data["split"], split_params=pipeline_data["split_params"])
        line.trained = True
        return line
    line = pipeline({'model' : model}, validation_split=pipeline_data["split"], split_params=pipeline_data["split_params"])
    line.trained = True
    return line
    

def load_pipeline(pipeline_file : str =None, model_file : str =None):
    if pipeline_file is None or model_file is None:
        raise InvalidValueError("Invalid file_name `None`")
    
    if not(pipeline_file.endswith(".prsl")):
        raise Error("Invalid pipeline file, pipeline extension should be `.prsl`")
    
    if not(model_file.endswith(".rslr")) and not(model_file.endswith(".rslc")):
        raise Error("model file extension should be `.rslr` or `.rslc`")
    
    with open(pipeline_file, "rb") as f:
        compressed = f.read()

    json_bytes = gzip.decompress(compressed)

    pipeline_data = json.loads(json_bytes.decode("utf-8"))

    if "rslearn_compressed" not in pipeline_data:
        raise Error("Invalid compressed file. giving file is not compressed by rslearn.")

    if "pipeline" not in pipeline_data:
        raise Error("Invalid Pipeline file.")

    with open(model_file, "rb") as f:
        compressed = f.read()

    json_bytes = gzip.decompress(compressed)

    model_data = json.loads(json_bytes.decode("utf-8"))

    if "rslearn_compressed" not in model_data:
        raise Error("Invalid compressed file. giving file is not compressed by rslearn.")
    if not(model_data["model"].endswith(str(pipeline_data["model_id"]))):
        raise Error("Invalid Model file, model file is not trained by pipeline.")
    
    # loading pipeline
   
    line = pipeline_load_helper(pipeline_data=pipeline_data, model_file=model_file)
    return line