import enum
import json
from http import HTTPStatus
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch.cuda
import yaml
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel, Field
from ray import serve
from ray.serve import Application
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer, pipeline

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
        return str(obj)


class InferenceRequest(BaseModel):
    args: List[Any] = Field(default=[])
    kwargs: Dict[str, Any] = Field(default={})


class Library(enum.Enum):
    transformers = "transformers"
    sentence_transformers = "sentence-transformers"


@serve.deployment
@serve.ingress(app)
class TransformersModelDeployment:
    def __init__(self, model_path: str, task: str, trust_remote_code: bool):
        self.model_path = model_path
        self.task = task
        self.trust_remote_code = trust_remote_code
        self.pipe = pipeline(
            task=self.task,
            model=self.model_path,
            trust_remote_code=self.trust_remote_code,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

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
class MiniCPMDeployment:
    def __init__(self, model_path: str, task: str, trust_remote_code: bool):
        self.model_path = model_path
        self.task = task
        self.trust_remote_code = trust_remote_code

        model = AutoModel.from_pretrained(
            model_path, trust_remote_code=trust_remote_code
        )
        model = model.to(device="cuda" if torch.cuda.is_available() else "cpu")
        self.model = model

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )

    @app.post("/image_infer")
    async def image_infer(
        self,
        # Accept args and kwargs as JSON:
        json_args: str = Form(...),
        json_kwargs: str = Form(...),
        image_file: UploadFile = File(...),
    ):
        # Deserialize the JSON string into a list of dictionaries
        try:
            args: list = json.loads(json_args)  # Convert the JSON string to a list
            kwargs: dict = json.loads(json_kwargs)
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST,
                detail="Invalid JSON format for messages",
            )

        img = Image.open(image_file.file).convert("RGB")
        kwargs["image"] = img

        print(f"{args=}")
        print(f"{kwargs=}")
        return self.model.chat(*args, **kwargs, image=img)

    @app.get("/model_config")
    def model_config(self):
        return numpy_to_std(self.model.config.__dict__)

    @app.get("/model_device")
    def model_device(self) -> str:
        return str(self.model.device)


@serve.deployment
@serve.ingress(app)
class SentenceTransformersModelDeployment:
    def __init__(self, model_path: str, trust_remote_code: bool):
        self.model_path = model_path
        self.trust_remote_code = trust_remote_code
        self.model = SentenceTransformer(
            self.model_path,
            trust_remote_code=self.trust_remote_code,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

    @app.get("/model_device")
    def model_device(self) -> str:
        return str(self.model.device)

    @app.post("/infer")
    def infer(self, inference_request: InferenceRequest) -> dict:
        args = inference_request.args
        kwargs = inference_request.kwargs
        return {"result": numpy_to_std(self.model.encode(*args, **kwargs))}


def deployment(args) -> Application:
    config = yaml.safe_load(open(Path(args["model_path"]) / "config.json", "r"))
    if (
        args["library"] == "transformers"
        and args["task"] == "visual-question-answering"
        and "auto_map" in config
        and config["auto_map"]["AutoConfig"].endswith("MiniCPMVConfig")
    ):
        return MiniCPMDeployment.bind(  # type: ignore[attr-defined]
            args["model_path"], args["task"], args["trust_remote_code"]
        )
    elif args["library"] == "transformers":
        return TransformersModelDeployment.bind(  # type: ignore[attr-defined]
            args["model_path"], args["task"], args["trust_remote_code"]
        )
    elif args["library"] == "sentence-transformers":
        return SentenceTransformersModelDeployment.bind(  # type: ignore[attr-defined]
            args["model_path"], args["trust_remote_code"]
        )
    else:
        raise ValueError(f"Library '{args['library']}' not supported")
