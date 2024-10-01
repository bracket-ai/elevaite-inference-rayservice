import torch.cuda
from fastapi import FastAPI
from ray import serve
from ray.serve import Application
from transformers import pipeline

from utils import InferenceRequest, numpy_to_std

web_app = FastAPI()


@serve.deployment
@serve.ingress(web_app)
class TransformersModelDeployment:
    def __init__(self, model_path: str, task: str, trust_remote_code: bool):
        self.model_path = model_path
        self.task = task
        self.trust_remote_code = trust_remote_code
        self.pipe = pipeline(
            task=self.task,
            model=self.model_path,
            trust_remote_code=self.trust_remote_code,
            low_mem_cpu_usage=True,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

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


def app_builder(args: dict) -> Application:
    return TransformersModelDeployment.bind(  # type: ignore[attr-defined]
        args["model_path"], args["task"], args["trust_remote_code"]
    )
