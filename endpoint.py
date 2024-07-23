import os
from pathlib import Path
from typing import Any, List, Dict

import numpy as np
import torch.cuda
from fastapi import FastAPI
from pydantic import BaseModel
from ray import serve

# These imports are used only for type hints:
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

LOCAL_MODEL_PATH = str(Path("model").absolute())


class InferenceRequest(BaseModel):
    args: List[Any]
    kwargs: Dict[str, Any]


@serve.deployment
@serve.ingress(app)
class ModelDeployment:
    def __init__(self):
        self.pipe = pipeline(
            task=os.getenv("TASK", "token-classification"),
            model=os.getenv("MODEL_PATH", "/model"),
            trust_remote_code=bool(int(os.getenv("TRUST_REMOTE_CODE", "0"))),
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
