import json
import logging
from http import HTTPStatus

import torch.cuda
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image
from ray import serve
from ray.serve import Application
from transformers import AutoProcessor, MllamaForConditionalGeneration

from utils import dtype_mapping, numpy_to_std

logger = logging.getLogger("ray.serve")

web_app = FastAPI()


@serve.deployment
@serve.ingress(web_app)
class LlamaVisionDeployment:
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
            "device_map": device,
        }

        model = MllamaForConditionalGeneration.from_pretrained(**model_args)
        model = model.to(device)
        model = model.eval()
        self.model = model

        self.processor = AutoProcessor.from_pretrained(model_path)

    def _clear_cache(self):
        try:
            if str(self.model.device) == "cuda":
                torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"Internal Server Error w/ clearing cache: {e}", exc_info=True)
            raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR)

    @web_app.post("/image_infer")
    async def image_infer(
        self,
        image_file: UploadFile = File(...),
        json_messages: str = Form(...),
        json_kwargs: str = Form(...),
    ):
        self._clear_cache()

        try:
            messages: list = json.loads(json_messages)
            kwargs: dict = json.loads(json_kwargs)
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST,
                detail="Invalid JSON format for messages",
            )

        try:
            image = Image.open(image_file.file)
        except Exception as e:
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST,
                detail=f"Invalid image input: {e}",
            )

        try:
            input_text = self.processor.apply_chat_template(
                messages, add_generation_prompt=True
            )
            inputs = self.processor(
                image, input_text, add_special_tokens=False, return_tensors="pt"
            ).to(self.model.device)

            with torch.no_grad():
                output = self.model.generate(**inputs, **kwargs)
                response = self.processor.decode(output[0])
                return {"response": response}

        except Exception as e:
            self._clear_cache()
            logger.error(f"Internal Server Error: {e}", exc_info=True)
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
    return LlamaVisionDeployment.bind(  # type: ignore[attr-defined]
        args["model_path"],
        args["task"],
        args["trust_remote_code"],
        args["device"],
        args["torch_dtype"],
    )