from http import HTTPStatus

import torch.cuda
from fastapi import FastAPI, HTTPException
from ray import serve
from ray.serve import Application
from sentence_transformers import SentenceTransformer

from utils import InferenceRequest, dtype_mapping, numpy_to_std

web_app = FastAPI()


@serve.deployment
@serve.ingress(web_app)
class SentenceTransformersModelDeployment:
    def __init__(
        self,
        model_path: str,
        trust_remote_code: bool,
        device: str = "auto",
        torch_dtype: str | None = None,
    ):
        self.model_path = model_path
        self.trust_remote_code = trust_remote_code
        self.device = device
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

        model_args = {
            "model_name_or_path": self.model_path,
            "trust_remote_code": self.trust_remote_code,
            "device": device,
        }

        model_kwargs = {}
        if self.torch_dtype:
            model_kwargs["torch_dtype"] = dtype_mapping.get(
                self.torch_dtype.lower(), None
            )

        if model_kwargs:
            model_args["model_kwargs"] = model_kwargs

        self.model = SentenceTransformer(**model_args)
        self.model = self.model.eval()

    @web_app.get("/model_device")
    def model_device(self) -> str:
        return str(self.model.device)

    @web_app.post("/infer")
    def infer(self, inference_request: InferenceRequest) -> dict:
        args = inference_request.args
        kwargs = inference_request.kwargs
        try:
            with torch.no_grad():
                return {"result": numpy_to_std(self.model.encode(*args, **kwargs))}
        except Exception as e:
            raise HTTPException(
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail=str(e)
            )
        finally:
            del args, kwargs
            if str(self.model.device) == "cuda":
                torch.cuda.empty_cache()  # Clear cache after processing

    @web_app.get("/get_num_threads")
    def get_num_threads(self):
        import os

        return {
            "cpu_count": os.cpu_count(),
            "num_threads": torch.get_num_threads(),
            "ray_omp_num_threads": os.environ.get("OMP_NUM_THREADS", None),
        }


def app_builder(args: dict) -> Application:
    return SentenceTransformersModelDeployment.bind(  # type: ignore[attr-defined]
        args["model_path"],
        args["trust_remote_code"],
        args["device"],
        args["torch_dtype"],
    )
