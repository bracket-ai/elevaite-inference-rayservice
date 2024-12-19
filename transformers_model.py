import logging
from http import HTTPStatus

import torch.cuda
from fastapi import FastAPI, HTTPException
from ray import serve
from ray.serve import Application
from transformers import pipeline

from utils import InferenceRequest, dtype_mapping, numpy_to_std

logger = logging.getLogger("ray.serve")

web_app = FastAPI()


@serve.deployment(health_check_period_s=30)
@serve.ingress(web_app)
class TransformersModelDeployment:
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

        pipe_kwargs = {
            "task": self.task,
            "model": self.model_path,
            "trust_remote_code": self.trust_remote_code,
            "device": device,
        }

        if torch_dtype:
            pipe_kwargs["torch_dtype"] = dtype_mapping.get(torch_dtype.lower())

        # No need to call .eval() here, since pipeline does it for us
        # https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/base.py#L287-L290
        self.pipe = pipeline(**pipe_kwargs)

    def _clear_cache(self):
        if str(self.pipe.device) == "cuda":
            torch.cuda.empty_cache()

    @web_app.get("/model_device")
    def model_device(self) -> str:
        return str(self.pipe.device)

    @web_app.post("/infer")
    def infer(self, inference_request: InferenceRequest) -> dict:
        """
        **Request Format:**
        ```json
        {
            "args": ["Help me write a poem that rhymes"],
            "kwargs": {"do_sample": false, "max_new_tokens": 50}
        }
        ```

        **Batch Processing:**
        ```json
        {
            "args": [["Write a haiku", "Write a limerick"]],
            "kwargs": {"do_sample": false, "max_new_tokens": 50}
        }
        ```

        **Example Python code:**
        ```python
        import requests

        url = "<URL>/<model_id>/infer"

        # Single text generation
        payload = {
            "args": ["Help me write a poem that rhymes"],
            "kwargs": {
                "do_sample": False,
                "max_new_tokens": 50
            }
        }

        # Or batch text generation
        batch_payload = {
            "args": [["Write a haiku", "Write a limerick"]],
            "kwargs": {
                "do_sample": False,
                "max_new_tokens": 50
            }
        }

        headers = {"Content-Type": "application/json"}

        # Basic authentication credentials
        username = "your_username"
        password = "your_password"

        response = requests.post(
            url,
            json=payload,  # or batch_payload
            headers=headers,
            auth=(username, password),
        )
        result = response.json()
        ```

        **Example curl commands:**

        Single generation:
        ```bash
        curl -X POST "<URL>/<model_id>/infer" \
        -H "Content-Type: application/json" \
        -u <username>:<password> \
        -d '{
            "args": ["Help me write a poem that rhymes"],
            "kwargs": {"do_sample": false, "max_new_tokens": 50}
        }'
        ```

        Batch generation:
        ```bash
        curl -X POST "<URL>/<model_id>/infer" \
        -H "Content-Type: application/json" \
        -u <username>:<password> \
        -d '{
            "args": [["Write a haiku", "Write a limerick"]],
            "kwargs": {"do_sample": false, "max_new_tokens": 50}
        }'
        ```
        """

        args = inference_request.args
        kwargs = inference_request.kwargs

        try:
            self._clear_cache()
            with torch.no_grad():
                result = self.pipe(*args, **kwargs)
            return {"result": numpy_to_std(result)}
        except Exception as e:
            self._clear_cache()
            logger.error(f"Internal Server Error: {e}", exc_info=True)
            raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR)
        finally:
            self._clear_cache()

    @web_app.get("/model_config")
    def model_config(self):
        return numpy_to_std(self.pipe.model.config.__dict__)

    @web_app.get("/health")
    def check_health(self):
        # We only run this for text-generation tasks at the moment
        # TODO: Add support for other tasks
        if self.task != "text-generation":
            pass

        try:
            self._clear_cache()

            # Basic inference test
            with torch.no_grad():
                self.pipe("Is this thing on?", max_new_tokens=10)

            logger.info("Health check passed")
            return {"status": "healthy"}

        except Exception as e:
            self._clear_cache()
            logger.error(f"Health check failed: {e}", exc_info=True)
            raise HTTPException(
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                detail="Endpoint is unhealthy. Basic model.pipe() call failed.",
            )
        finally:
            self._clear_cache()


def app_builder(args: dict) -> Application:
    return TransformersModelDeployment.bind(  # type: ignore[attr-defined]
        args["model_path"],
        args["task"],
        args["trust_remote_code"],
        args["device"],
        args["torch_dtype"],
    )
