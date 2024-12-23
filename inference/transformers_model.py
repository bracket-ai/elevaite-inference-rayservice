import logging
from http import HTTPStatus

import torch.cuda
from fastapi import FastAPI, HTTPException
from ray import serve
from ray.serve import Application
from transformers import pipeline

from inference.utils import InferenceRequest, dtype_mapping, numpy_to_std

SUPPORTED_HEALTH_CHECK_TASKS = {
    "feature-extraction",
    "summarization",
    "text2text-generation",
    "text-classification",
    "text-generation",
    "token-classification",
    "zero-shot-classification",
}


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
        Perform inference using the model pipeline. Supports both text generation and chat completion
        with single, serial, and batch processing options.

        **Request Format:**
        The endpoint expects a JSON request with `args` (input data) and `kwargs` (generation parameters):
        ```json
        {
            "args": [...],     # Input data
            "kwargs": {        # Generation parameters
                "do_sample": false,
                "max_new_tokens": 50,
                "batch_size": 4  # Optional, for batch processing
            }
        }
        ```

        **Examples:**

        1. Single Text Generation:
        ```json
        {
            "args": ["Write a poem"],
            "kwargs": {
                "do_sample": false,
                "max_new_tokens": 50
            }
        }
        ```

        2. Multiple Text Generation (Serial/Batch):
        ```json
        {
            "args": [["Write a haiku", "Write a song"]],
            "kwargs": {
                "do_sample": false,
                "max_new_tokens": 50,
                "batch_size": 4  # Optional, enables batch processing
            }
        }
        ```

        3. Single Chat Completion:
        ```json
        {
            "args": [[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": "Hello!"}
            ]],
            "kwargs": {
                "do_sample": false,
                "max_new_tokens": 50
            }
        }
        ```

        4. Multiple Chat Completion (Serial/Batch):
        ```json
        {
            "args": [[
                [
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": "Hello!"}
                ],
                [
                    {"role": "system", "content": "You are a pirate"},
                    {"role": "user", "content": "Hello!"}
                ]
            ]],
            "kwargs": {
                "do_sample": false,
                "max_new_tokens": 50,
                "batch_size": 2  # Optional, enables batch processing
            }
        }
        ```

        **Notes:**
        - Chat inputs must be properly nested within lists
        - Batch processing is controlled via the optional `batch_size` parameter
        - Memory usage scales with batch size

        **Example Python request:**
        ```python
        import requests

        url = "http://<URL>/<endpoint>/infer"
        headers = {"Content-Type": "application/json"}

        # Replace with any request JSON from examples above
        request = {
            "args": ["Write a poem"],
            "kwargs": {
                "do_sample": False,
                "max_new_tokens": 50
            }
        }

        response = requests.post(url, json=request, headers=headers)
        result = response.json()
        ```

        **Example curl request:**
        ```bash
        curl -X POST "http://<URL>/<endpoint>/infer" \
        -H "Content-Type: application/json" \
        -d '<REPLACE_WITH_REQUEST_JSON_FROM_EXAMPLES_ABOVE>'
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

        try:
            # cache is only cleared if model is on CUDA, where we are memory-
            # constrained.
            self._clear_cache()

            # For pipelines that don't support the most basic inference test,
            # we don't currently support health checking them
            # FIXME: Add support for health checking these tasks.
            # Will require matching the more complex call signature of these tasks.
            if self.task not in SUPPORTED_HEALTH_CHECK_TASKS:
                return {"warning": f"Health check not supported for task {self.task}"}

            # Basic inference test
            # If this errors, the health check will fail.
            with torch.no_grad():
                if self.task == "text-generation":
                    self.pipe("Is this thing on?", max_new_tokens=10)
                else:
                    self.pipe("Is this thing on?")

            logger.info("Health check passed")
            return {"status": "healthy"}

        except Exception as e:
            self._clear_cache()
            logger.error(f"Health check failed: {e}", exc_info=True)
            raise HTTPException(
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                detail="Endpoint is unhealthy. Basic inference call failed.",
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
