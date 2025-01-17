import logging
from http import HTTPStatus

import torch.cuda
from fastapi import FastAPI, HTTPException
from ray import serve
from ray.serve import Application
from sentence_transformers import SentenceTransformer

from inference.utils import InferenceRequest, dtype_mapping, numpy_to_std

logger = logging.getLogger("ray.serve")

web_app = FastAPI()


@serve.deployment(health_check_period_s=30)
@serve.ingress(web_app)
class SentenceTransformersModelDeployment:
    def __init__(
        self,
        model_path: str,
        trust_remote_code: bool,
        device: str = "auto",
        torch_dtype: str = "float32",
    ):
        self.model_path = model_path
        self.trust_remote_code = trust_remote_code
        self.torch_dtype = torch_dtype

        if device not in ["cuda", "cpu"]:
            raise ValueError("device must be one of 'cuda' or 'cpu'")

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is not available. Please check "
                "for available resources."
            )

        sentence_transformers_kwargs = {
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
            sentence_transformers_kwargs["model_kwargs"] = model_kwargs

        self.model = SentenceTransformer(**sentence_transformers_kwargs)
        self.model = self.model.eval()

    @web_app.get("/model_device")
    def model_device(self) -> str:
        return str(self.model.device)

    @web_app.post("/infer")
    def infer(self, inference_request: InferenceRequest) -> dict:
        """
        **Request Format:**
        ```json
        {
            "args": ["This is a sentence to embed"],
            "kwargs": {}
        }
        ```

        **Batch Processing:**
        ```json
        {
            "args": [["First sentence", "Second sentence"]],
            "kwargs": {}
        }
        ```

        **Example Python code:**
        ```python
        import requests

        url = "<URL>/<model_id>/infer"
        payload = {
            "args": ["This is a sentence to embed"],
            "kwargs": {}
        }
        headers = {"Content-Type": "application/json"}

        # Basic authentication credentials
        username = "your_username"
        password = "your_password"

        response = requests.post(
            url,
            json=payload,
            headers=headers,
            auth=(username, password),
        )
        result = response.json()
        ```

        **Example curl commands:**

        Single text embedding:
        ```bash
        curl -X POST "<URL>/<model_id>/infer" \\
        -H "Content-Type: application/json" \\
        -d '{"args": ["This is a sentence to embed"], "kwargs": {}}'
        ```

        Batch text embedding:
        ```bash
        curl -X POST "<URL>/<model_id>/infer" \\
        -H "Content-Type: application/json" \\
        -d '{"args": [["First sentence", "Second sentence"]], "kwargs": {}}'
        ```
        """
        args = inference_request.args
        kwargs = inference_request.kwargs
        try:
            with torch.no_grad():
                return {"result": numpy_to_std(self.model.encode(*args, **kwargs))}
        except Exception as e:
            logger.error(f"Internal Server Error: {e}", exc_info=True)
            raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR)

    @web_app.get("/health")
    def check_health(self):
        """Health check that verifies basic model functionality."""
        try:
            # Basic inference test
            with torch.no_grad():
                self.model.encode("Is this thing on?")

            logger.info("Health check passed")
            return {"status": "healthy"}

        except Exception as e:
            logger.error(f"Health check failed: {e}", exc_info=True)
            raise HTTPException(
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                detail="Endpoint is unhealthy. Basic model.encode() call failed.",
            )


def app_builder(args: dict) -> Application:
    return SentenceTransformersModelDeployment.bind(  # type: ignore[attr-defined]
        args["model_path"],
        args["trust_remote_code"],
        args["device"],
        args["torch_dtype"],
    )
