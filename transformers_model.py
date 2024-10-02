import torch.cuda
from fastapi import FastAPI
from ray import serve
from ray.serve import Application
from transformers import pipeline

from utils import InferenceRequest, dtype_mapping, numpy_to_std

web_app = FastAPI()


@serve.deployment
@serve.ingress(web_app)
class TransformersModelDeployment:
    def __init__(
        self,
        model_path: str,
        task: str,
        trust_remote_code: bool,
        torch_dtype: str | None = None,
    ):
        self.model_path = model_path
        self.task = task
        self.trust_remote_code = trust_remote_code
        self.torch_dtype = torch_dtype
        pipe_kwargs = {
            "task": self.task,
            "model": self.model_path,
            "trust_remote_code": self.trust_remote_code,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "model_kwargs": {"low_cpu_mem_usage": True},
        }

        if torch_dtype:
            pipe_kwargs["torch_dtype"] = dtype_mapping.get(torch_dtype.lower(), None)

        # No need to call .eval() here, since pipeline does it for us
        # https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/base.py#L287-L290
        self.pipe = pipeline(**pipe_kwargs)

    @web_app.get("/model_device")
    def model_device(self) -> str:
        return str(self.pipe.device)

    @web_app.post("/infer")
    def infer(self, inference_request: InferenceRequest) -> dict:
        args = inference_request.args
        kwargs = inference_request.kwargs
        return {"result": numpy_to_std(self.pipe(*args, **kwargs))}

    @web_app.get("/model_config")
    def model_config(self):
        return numpy_to_std(self.pipe.model.config.__dict__)

    @web_app.get("/device_map")
    def device_map(self) -> dict:
        return numpy_to_std(self.pipe.model.hf_device_map)


def app_builder(args: dict) -> Application:
    return TransformersModelDeployment.bind(  # type: ignore[attr-defined]
        args["model_path"], args["task"], args["trust_remote_code"], args["torch_dtype"]
    )
