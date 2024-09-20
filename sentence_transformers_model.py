import torch.cuda
from fastapi import FastAPI
from ray import serve
from ray.serve import Application
from sentence_transformers import SentenceTransformer

from utils import InferenceRequest, numpy_to_std

web_app = FastAPI()


@serve.deployment
@serve.ingress(web_app)
class SentenceTransformersModelDeployment:
    def __init__(self, model_path: str, trust_remote_code: bool):
        self.model_path = model_path
        self.trust_remote_code = trust_remote_code
        self.model = SentenceTransformer(
            self.model_path,
            trust_remote_code=self.trust_remote_code,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

    @web_app.get("/model_device")
    def model_device(self) -> str:
        return str(self.model.device)

    @web_app.post("/infer")
    def infer(self, inference_request: InferenceRequest) -> dict:
        args = inference_request.args
        kwargs = inference_request.kwargs
        return {"result": numpy_to_std(self.model.encode(*args, **kwargs))}


def app_builder(args: dict) -> Application:
    return SentenceTransformersModelDeployment.bind(  # type: ignore[attr-defined]
        args["model_path"], args["trust_remote_code"]
    )
