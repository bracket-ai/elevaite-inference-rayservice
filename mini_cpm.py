import gc
import json
from http import HTTPStatus

import torch.cuda
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image
from ray import serve
from ray.serve import Application
from transformers import AutoModel, AutoTokenizer

from utils import dtype_mapping, numpy_to_std

web_app = FastAPI()


@serve.deployment
@serve.ingress(web_app)
class MiniCPMDeployment:
    def __init__(
        self,
        model_path: str,
        task: str,
        trust_remote_code: bool,
        device: str = "auto",
        torch_dtype: str | None = None,
    ):
        self.model_path = model_path
        self.task = task
        self.trust_remote_code = trust_remote_code
        self.torch_dtype = torch_dtype

        if device not in ["cuda", "auto", "cpu"]:
            raise ValueError("device must be one of 'auto', 'cuda', or 'cpu'")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is not available. Please check "
                "for available resources."
            )

        self.device = device

        model_args = {
            "pretrained_model_name_or_path": model_path,
            "trust_remote_code": trust_remote_code,
            "device": device,
        }

        if torch_dtype:
            model_args["torch_dtype"] = dtype_mapping.get(torch_dtype.lower(), None)

        model = AutoModel.from_pretrained(**model_args)

        model = model.eval()
        self.model = model

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )

    def _clear_cache(self):
        if str(self.model.device) == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    @web_app.post("/image_infer")
    async def image_infer(
        self,
        # Accept args and kwargs as JSON:
        image_files: list[UploadFile] = File([]),
        json_messages: str = Form(...),
        json_kwargs: str = Form(...),
    ):
        # Deserialize the JSON string into a list of dictionaries
        try:
            messages: list = json.loads(
                json_messages
            )  # Convert the JSON string to a list
            kwargs: dict = json.loads(json_kwargs)
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST,
                detail="Invalid JSON format for messages",
            )

        try:
            self._clear_cache()
            kwargs["tokenizer"] = self.tokenizer
            processed_messages = []
            for message in messages:
                processed_message = {**message}
                if type(message["content"]) is list:
                    processed_content = []
                    for item in message["content"]:
                        if type(item) is int:
                            if item not in range(len(image_files)):
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

            print(f"{kwargs=}")
            with torch.no_grad():
                return self.model.chat(**kwargs)
        except Exception as e:
            self._clear_cache()
            raise HTTPException(
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail=str(e)
            )
        finally:
            self._clear_cache()
            del kwargs

    @web_app.get("/model_config")
    def model_config(self):
        return numpy_to_std(self.model.config.__dict__)

    @web_app.get("/model_device")
    def model_device(self) -> str:
        return str(self.model.device)

    @web_app.get("/get_num_threads")
    def get_num_threads(self):
        import os

        return {
            "cpu_count": os.cpu_count(),
            "num_threads": torch.get_num_threads(),
            "ray_omp_num_threads": os.environ.get("OMP_NUM_THREADS", None),
        }


def app_builder(args: dict) -> Application:
    return MiniCPMDeployment.bind(  # type: ignore[attr-defined]
        args["model_path"],
        args["task"],
        args["trust_remote_code"],
        args["device"],
        args["torch_dtype"],
    )
