import enum
import json
import logging
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
from torch.profiler import profile, ProfilerActivity
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
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        self.logger = logging.getLogger(__name__)

    @app.get("/model_device")
    def model_device(self) -> str:
        return str(self.pipe.device)

    @app.post("/infer")
    def infer(self, inference_request: InferenceRequest) -> dict:
        args = inference_request.args
        kwargs = inference_request.kwargs

        self.logger.info(f"Received inference request with args: {args}, kwargs: {kwargs}")

        # Profiling block
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
            result = self.pipe(*args, **kwargs)
        
        # Log the profiling results
        self.logger.info(f"Profiling results:\n{prof.key_averages().table(sort_by="cuda_time_total", row_limit=10)}")

        return {"result": numpy_to_std(result)}

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

        # Setup logging
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        self.logger = logging.getLogger(__name__)

        self.logger.info(f"Initializing MiniCPMDeployment with model_path: {model_path}, task: {task}, trust_remote_code: {trust_remote_code}")

        model = AutoModel.from_pretrained(
            model_path, trust_remote_code=trust_remote_code
        )
        model = model.to(device="cuda" if torch.cuda.is_available() else "cpu")
        self.model = model

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )

        self.logger.info(f"Model and tokenizer loaded successfully. Model device: {self.model.device}")

    @app.post("/image_infer")
    async def image_infer(
        self,
        # Accept args and kwargs as JSON:
        image_files: list[UploadFile] = File([]),
        json_messages: str = Form(...),
        json_kwargs: str = Form(...),
    ):
        self.logger.info("Received image_infer request")
        # Deserialize the JSON string into a list of dictionaries
        try:
            messages: list = json.loads(
                json_messages
            )  # Convert the JSON string to a list
            kwargs: dict = json.loads(json_kwargs)
            self.logger.debug(f"Deserialized messages: {messages}")
            self.logger.debug(f"Deserialized kwargs: {kwargs}")
        except json.JSONDecodeError:
            self.logger.error("Invalid JSON format for messages")
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST,
                detail="Invalid JSON format for messages",
            )

        kwargs["tokenizer"] = self.tokenizer
        processed_messages = []
        for message in messages:
            processed_message = {**message}
            if type(message["content"]) is list:
                processed_content = []
                for item in message["content"]:
                    if type(item) is int:
                        if item not in range(len(image_files)):
                            self.logger.error(f"Invalid image index: {item}")
                            raise HTTPException(
                                status_code=HTTPStatus.BAD_REQUEST,
                                detail="Image indices must be 0-based, in range of total uploaded image files",
                            )
                        processed_content.append(
                            Image.open(image_files[item].file).convert("RGB")
                        )
                    else:
                        processed_content.append(item)
                processed_message["content"] = processed_content
            processed_messages.append(processed_message)
        kwargs["msgs"] = processed_messages
        kwargs["image"] = None

        self.logger.info(f"Processed kwargs: {kwargs}")
        result = self.model.chat(**kwargs)
        self.logger.info("Chat inference completed successfully")
        return result

    @app.get("/model_config")
    def model_config(self):
        self.logger.info("Retrieving model config")
        return numpy_to_std(self.model.config.__dict__)

    @app.get("/model_device")
    def model_device(self) -> str:
        self.logger.info("Retrieving model device")
        return str(self.model.device)


@serve.deployment
@serve.ingress(app)
class SentenceTransformersModelDeployment:
    def __init__(self, model_path: str, trust_remote_code: bool):
        self.model_path = model_path
        self.trust_remote_code = trust_remote_code

        # Setup logging
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        self.logger = logging.getLogger(__name__)

        self.logger.info(f"Initializing SentenceTransformersModelDeployment with model_path: {model_path}, trust_remote_code: {trust_remote_code}")

        self.model = SentenceTransformer(
            self.model_path,
            trust_remote_code=self.trust_remote_code,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        self.logger.info(f"Model loaded successfully. Model device: {self.model.device}")

    @app.get("/model_device")
    def model_device(self) -> str:
        self.logger.info("Retrieving model device")
        return str(self.model.device)

    @app.post("/infer")
    def infer(self, inference_request: InferenceRequest) -> dict:
        self.logger.info("Received inference request")
        args = inference_request.args
        kwargs = inference_request.kwargs
        self.logger.debug(f"Inference args: {args}")
        self.logger.debug(f"Inference kwargs: {kwargs}")
        result = self.model.encode(*args, **kwargs)
        self.logger.info("Inference completed successfully")
        return {"result": numpy_to_std(result)}


def deployment(args) -> Application:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    logger.info(f"Deploying model with args: {args}")

    config = yaml.safe_load(open(Path(args["model_path"]) / "config.json", "r"))
    if config.get("auto_map", {}).get("AutoConfig", "").endswith("MiniCPMVConfig"):
        logger.info("Deploying MiniCPMDeployment")
        return MiniCPMDeployment.bind(  # type: ignore[attr-defined]
            args["model_path"], args["task"], args["trust_remote_code"]
        )
    elif args["library"] == "transformers":
        logger.info("Deploying TransformersModelDeployment")
        return TransformersModelDeployment.bind(  # type: ignore[attr-defined]
            args["model_path"], args["task"], args["trust_remote_code"]
        )
    elif args["library"] == "sentence-transformers":
        logger.info("Deploying SentenceTransformersModelDeployment")
        return SentenceTransformersModelDeployment.bind(  # type: ignore[attr-defined]
            args["model_path"], args["trust_remote_code"]
        )
    else:
        logger.error(f"Unsupported library: {args['library']}")
        raise ValueError(f"Library '{args['library']}' not supported")
