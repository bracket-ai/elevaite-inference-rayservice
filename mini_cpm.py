import json
from http import HTTPStatus

import torch.cuda
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image
from ray import serve
from ray.serve import Application
from transformers import AutoModel, AutoTokenizer

from utils import numpy_to_std, dtype_mapping

web_app = FastAPI()


@serve.deployment
@serve.ingress(web_app)
class MiniCPMDeployment:
    def __init__(self, model_path: str, task: str, trust_remote_code: bool, torch_dtype: str = None):
        self.model_path = model_path
        self.task = task
        self.trust_remote_code = trust_remote_code

        model_args = {
            "model_name_or_path": self.model_path,
            "trust_remote_code": self.trust_remote_code,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        }

        if torch_dtype:
            model_args["torch_dtype"] = dtype_mapping.get(torch_dtype.lower(), None)

        model = AutoModel.from_pretrained(
            **model_args
        )
        model = model.to(device="cuda" if torch.cuda.is_available() else "cpu")
        model = model.eval()
        self.model = model

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )

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

    @web_app.get("/model_config")
    def model_config(self):
        return numpy_to_std(self.model.config.__dict__)

    @web_app.get("/model_device")
    def model_device(self) -> str:
        return str(self.model.device)
    
    @web_app.get("/device_map")
    def device_map(self) -> dict:
        return numpy_to_std(self.model.hf_device_map)


def app_builder(args: dict) -> Application:
    return MiniCPMDeployment.bind(  # type: ignore[attr-defined]
        args["model_path"], args["task"], args["trust_remote_code"]
    )
