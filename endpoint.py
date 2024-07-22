import os
import shutil
from pathlib import Path
from typing import Any, List, Dict

import elevaite_file_client
import numpy as np
import torch.cuda
from fastapi import FastAPI
from pydantic import BaseModel
from ray import serve

# These imports are used only for type hints:
from starlette.requests import Request
from transformers import pipeline

app = FastAPI()


def numpy_to_std(obj):
    """Convert all objects in dict (recursively) from numpy types to vanilla
    Python types."""
    if isinstance(obj, list):
        new_obj = []
        for item in obj:
            new_obj.append(numpy_to_std(item))
        return new_obj
    elif isinstance(obj, dict):
        new_obj = {}
        for key, value in obj.items():
            new_obj[key] = numpy_to_std(value)
        return new_obj
    elif type(obj) in (int, float, str, bool) or obj is None:
        return obj
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, torch.dtype):
        return str(obj)
    else:
        raise TypeError(f"Could not serialize evaluation object: {obj}")

LOCAL_MODEL_PATH = "model"
MODEL_URL = os.getenv("MODEL_URL")


class InferenceRequest(BaseModel):
    args: List[Any]
    kwargs: Dict[str, Any]


@serve.deployment
@serve.ingress(app)
class ModelDeployment:
    def __init__(self):
        if Path(LOCAL_MODEL_PATH).exists():
            shutil.rmtree(LOCAL_MODEL_PATH)
        elevaite_file_client.download_directory(MODEL_URL, LOCAL_MODEL_PATH)
        self.pipe = pipeline(
            os.getenv("TASK"),
            model=LOCAL_MODEL_PATH,
            trust_remote_code=bool(int(os.getenv("ELEVAITE_TRUST_REMOTE_CODE", "0"))),
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

    @app.post("/infer")
    def infer(self, inference_request: InferenceRequest) -> dict:
        args = inference_request.args
        kwargs = inference_request.kwargs
        return {"result": numpy_to_std(self.pipe(*args, **kwargs))}

    @app.get("/model_config")
    def model_config(self):
        return numpy_to_std(self.pipe.model.config.__dict__)

deployment = ModelDeployment.bind()
