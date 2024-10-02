import torch.cuda
from fastapi import FastAPI
from ray import serve
from ray.serve import Application
from sentence_transformers import SentenceTransformer

from utils import InferenceRequest, dtype_mapping, numpy_to_std

web_app = FastAPI()


@serve.deployment
@serve.ingress(web_app)
class SentenceTransformersModelDeployment:
    def __init__(
        self, model_path: str, trust_remote_code: bool, torch_dtype: str | None = None
    ):
        self.model_path = model_path
        self.trust_remote_code = trust_remote_code
        self.torch_dtype = torch_dtype

        model_args = {
            "model_name_or_path": self.model_path,
            "trust_remote_code": self.trust_remote_code,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        }

        model_kwargs = {"low_cpu_mem_usage": True}
        if self.torch_dtype:
            model_kwargs["torch_dtype"] = dtype_mapping.get(
                self.torch_dtype.lower(), None
            )

        model_args.update(model_kwargs)
        self.model = SentenceTransformer(**model_args, **model_kwargs)
        self.model = self.model.eval()

    @web_app.get("/model_device")
    def model_device(self) -> str:
        return str(self.model.device)

    @web_app.post("/infer")
    def infer(self, inference_request: InferenceRequest) -> dict:
        args = inference_request.args
        kwargs = inference_request.kwargs
        with torch.no_grad():
            return {"result": numpy_to_std(self.model.encode(*args, **kwargs))}


def app_builder(args: dict) -> Application:
    return SentenceTransformersModelDeployment.bind(  # type: ignore[attr-defined]
        args["model_path"], args["trust_remote_code"]
    )
