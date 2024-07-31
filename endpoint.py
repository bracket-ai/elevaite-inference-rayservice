import enum
from pathlib import Path
from typing import Any, List, Dict

import numpy as np
import torch.cuda
from fastapi import FastAPI
from pydantic import BaseModel, Field
from ray import serve
from ray.serve import Application

from sentence_transformers import SentenceTransformer
from transformers import pipeline

app = FastAPI()


def numpy_to_std(obj):
    """Convert all objects in dict (recursively) from numpy types to vanilla
    Python types."""
    if isinstance(obj, list) or isinstance(obj, np.ndarray):
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
        raise TypeError(f"Could not serialize evaluation object: {type(obj)=} {obj}")


class InferenceRequest(BaseModel):
    args: List[Any] = Field(default=[])
    kwargs: Dict[str, Any] = Field(default={})


class Library(enum.Enum):
    transformers = "transformers"
    sentence_transformers = "sentence-transformers"


@serve.deployment
@serve.ingress(app)
class TransformersModelDeployment:
    def _refresh(self):
        self.pipe = pipeline(
            task=self.task,
            model=self.model_path,
            trust_remote_code=self.trust_remote_code,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

    def __init__(self, model_path: str, task: str, trust_remote_code: bool):
        self.model_path = model_path
        self.task = task
        self.trust_remote_code = trust_remote_code
        self._refresh()

    @app.post("/refresh_model")
    def refresh_model(self):
        self._refresh()

    @app.get("/model_device")
    def model_device(self) -> str:
        return str(self.pipe.device)

    @app.post("/infer")
    def infer(self, inference_request: InferenceRequest) -> dict:
        args = inference_request.args
        kwargs = inference_request.kwargs
        return {"result": numpy_to_std(self.pipe(*args, **kwargs))}

    @app.get("/model_config")
    def model_config(self):
        return numpy_to_std(self.pipe.model.config.__dict__)


@serve.deployment
@serve.ingress(app)
class SentenceTransformersModelDeployment:
    def _refresh(self):
        self.model = SentenceTransformer(
            self.model_path,
            trust_remote_code=self.trust_remote_code,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

    def __init__(self, model_path: str, trust_remote_code: bool):
        self.model_path = model_path
        self.trust_remote_code = trust_remote_code
        self._refresh()

    @app.post("/refresh_model")
    def refresh_model(self):
        self._refresh()

    @app.get("/model_device")
    def model_device(self) -> str:
        return str(self.model.device)

    @app.post("/infer")
    def infer(self, inference_request: InferenceRequest) -> dict:
        args = inference_request.args
        kwargs = inference_request.kwargs
        return {"result": numpy_to_std(self.model.encode(*args, **kwargs))}


def deployment(args) -> Application:
    if args["library"] == "transformers":
        return TransformersModelDeployment.bind(args["model_path"], args["task"], args["trust_remote_code"])
    elif args["library"] == "sentence-transformers":
        return SentenceTransformersModelDeployment.bind(args["model_path"], args["trust_remote_code"])
    else:
        raise ValueError(f"Library '{args['library']}' not supported")
