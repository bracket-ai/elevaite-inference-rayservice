import os
from typing import Any, List, Dict

import numpy as np
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
            if type(key) is not str:
                raise TypeError(
                    f"Dictionary contains invalid key {key!r}; {type(key)=}"
                )
            new_obj[key] = numpy_to_std(value)
        return new_obj
    elif type(obj) in (int, float, str):
        return obj
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    else:
        raise TypeError(f"Could not serialize evaluation object: {obj}")

model_name = os.getenv("MODEL_NAME", "dslim/bert-base-NER")


class InferenceRequest(BaseModel):
    args: List[Any]
    kwargs: Dict[str, Any]


@serve.deployment
@serve.ingress(app)
class ModelDeployment:
    def __init__(self):
        model_name = os.getenv("MODEL_NAME", "dslim/bert-base-NER")
        self.pipe = pipeline(model=model_name)

    @app.post("/infer")
    def infer(self, inference_request: InferenceRequest) -> dict:
        args = inference_request.args
        kwargs = inference_request.kwargs
        return {"result": numpy_to_std(self.pipe(*args, **kwargs))}

    @app.get("/model_config")
    def model_config(self):
        return self.pipe.model.config

deployment = ModelDeployment.bind()
