import json
import logging
from http import HTTPStatus

import torch.cuda
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image
from ray import serve
from ray.serve import Application
from transformers import AutoModel, AutoTokenizer

from utils import dtype_mapping, numpy_to_std

logger = logging.getLogger("ray.serve")

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
        torch_dtype: str = "float32",
    ):
        self.model_path = model_path
        self.task = task
        self.trust_remote_code = trust_remote_code
        self.torch_dtype = torch_dtype

        if device not in ["cuda", "cpu"]:
            raise ValueError("device must be one of 'cuda' or 'cpu'")

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is not available. Please check "
                "for available resources."
            )

        model_args = {
            "pretrained_model_name_or_path": model_path,
            "trust_remote_code": trust_remote_code,
            "torch_dtype": dtype_mapping.get(torch_dtype.lower()),
        }

        model = AutoModel.from_pretrained(**model_args)
        model = model.to(device)
        model = model.eval()
        self.model = model

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )

    def _clear_cache(self):
        if str(self.model.device) == "cuda":
            torch.cuda.empty_cache()

    @web_app.post("/image_infer")
    async def image_infer(
        self,
        image_files: list[UploadFile] = File([]),
        json_messages: str = Form(...),
        json_kwargs: str = Form(...),
    ):
        """
        **Request Format:**
        - `image_files`: Image files to process
            ```
            [image1.jpg, image2.jpg, ...]
            ```
        - `json_messages`: JSON array of messages in the format:
            ```json
            [
                {
                    "role": "user",
                    "content": [
                        "Look at these images:",
                        0,  # Reference to first image
                        1,  # Reference to second image
                        "What's the difference between them?"
                    ]
                }
            ]
            ```
        - `json_kwargs`: Additional parameters as JSON object
            ```json
            {"max_new_tokens": 50}
            ```

        **Example Python code:**
        ```python
        import requests
        import json

        URL = "<your_url>/model_id/image_infer"
        username = "your_username"
        password = "your_password"

        files = []
        for img_path in ['image_1.jpg', 'image_2.jpg']:
            with open(img_path, 'rb') as f:
                files.append(
                    ('image_files', (img_path, f.read(), 'image/jpeg'))
                )

        messages = [
            {
                "type": "text",
                "role": "user",
                "content": [
                    "Look at these images:", 0, 1, # indices of attached images to look at
                    "What is the difference between them?"
                ]
            }
        ]

        kwargs = {"max_new_tokens": 50}

        data = {
            'json_messages': json.dumps(messages),
            'json_kwargs': json.dumps(kwargs)
        }

        response = requests.post(
            URL,
            files=files,
            data=data,  # Use data for the JSON strings
            auth=(username, password),
            verify=False
        )

        result = response.json()
        ```

        **Example curl command:**
        ```bash
        curl -X 'POST' \\
            '<your_url>/model_id/image_infer' \\
            -H 'accept: application/json' \\
            -H 'Content-Type: multipart/form-data' \\
            -F 'image_files=@image_1.jpg;type=image/jpeg' \\
            -F 'image_files=@image_2.jpg;type=image/jpeg' \\
            -F 'json_messages=[{"type": "text", "role": "user", "content": ["Look at these images:", 0, 1, "What is the difference between them?"]}]' \\
            -F 'json_kwargs={}' \\
            -u '<username>:<password>'
        ```
        """
        try:
            messages: list = json.loads(json_messages)
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
            logger.error(f"Internal Server Error: {e}", exc_info=True)
            raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR)
        finally:
            self._clear_cache()

    @web_app.get("/health")
    def check_health(self):
        """Simple health check endpoint that tests basic model functionality"""
        try:
            self._clear_cache()
            kwargs = {
                "tokenizer": self.tokenizer,
                "msgs": [{"role": "user", "content": "Is this thing on?"}],
                "max_new_tokens": 10,
            }
            with torch.no_grad():
                return self.model.chat(**kwargs)
        except Exception as e:
            self._clear_cache()
            logger.error(f"Health check failed: {e}", exc_info=True)
            raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR)
        finally:
            self._clear_cache()

    @web_app.get("/model_config")
    def model_config(self):
        return numpy_to_std(self.model.config.__dict__)

    @web_app.get("/model_device")
    def model_device(self) -> str:
        return str(self.model.device)


def app_builder(args: dict) -> Application:
    return MiniCPMDeployment.bind(  # type: ignore[attr-defined]
        args["model_path"],
        args["task"],
        args["trust_remote_code"],
        args["device"],
        args["torch_dtype"],
    )
