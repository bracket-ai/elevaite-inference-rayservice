import json
import logging
from http import HTTPStatus
from typing import Any, List

import torch.cuda
from fastapi import FastAPI, HTTPException
from ray import serve
from ray.serve import Application
from sentence_transformers import SentenceTransformer

from inference.utils import (
    BatchableInferenceRequest,
    BatchingConfig,
    BatchingConfigUpdateRequest,
    BatchingConfigUpdateResponse,
    dtype_mapping,
    numpy_to_std,
)

logger = logging.getLogger("ray.serve")

web_app = FastAPI()


@serve.deployment(health_check_period_s=30, max_ongoing_requests=100)
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

        self.batching_enabled = True  # Always true since encode() handles batching

    @web_app.get("/model_device")
    def model_device(self) -> str:
        return str(self.model.device)

    @serve.batch(max_batch_size=32, batch_wait_timeout_s=0.1)
    async def _batch_infer(
        self, requests: List[BatchableInferenceRequest]
    ) -> List[dict]:
        try:
            logger.info(f"Starting batch call with {len(requests)} requests")

            # Group by kwargs since encode() supports different options
            current_group: List[Any] = []
            current_kwargs: str | None = None
            results: List[dict] = []

            for request in requests:
                # Handle None kwargs by converting to empty dict
                kwargs_to_serialize = (
                    request.kwargs if request.kwargs is not None else {}
                )
                kwargs_key = json.dumps(kwargs_to_serialize, sort_keys=True)

                if kwargs_key != current_kwargs and current_group:

                    if current_kwargs is None:
                        raise ValueError(
                            "current_kwargs should not be None at this point"
                        )

                    # Process current group
                    with torch.no_grad():
                        # returns a numpy.ndarray of shape (n_sentences, embedding_dim)
                        group_results = self.model.encode(
                            current_group,
                            batch_size=len(current_group),
                            **json.loads(current_kwargs),
                        )

                    if len(group_results.shape) == 2:  # 2D array (batch, embedding_dim)
                        results.extend(
                            {"result": numpy_to_std(row)} for row in group_results
                        )
                    else:  # Handle case where there's only one dimension
                        results.append({"result": numpy_to_std(group_results)})
                    current_group = []

                # args[0] is guaranteed to be string, dict, or list of dicts
                current_group.append(request.args[0])
                current_kwargs = kwargs_key

            # Process final group
            if current_group:
                if current_kwargs is None:
                    raise ValueError("current_kwargs should not be None at this point")

                with torch.no_grad():
                    group_results = self.model.encode(
                        current_group,
                        batch_size=len(current_group),
                        **json.loads(current_kwargs),
                    )
                if len(group_results.shape) == 2:  # 2D array (batch, embedding_dim)
                    results.extend(
                        {"result": numpy_to_std(row)} for row in group_results
                    )
                else:  # Handle case where there's only one dimension
                    results.append({"result": numpy_to_std(group_results)})

            return results

        except Exception as e:
            logger.error(f"Batch Inference Error: {e}", exc_info=True)
            raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR)

    @web_app.post("/infer")
    async def infer(self, inference_request: BatchableInferenceRequest) -> dict:
        """
        **Request Format:**
        ```json
        {
            "args": ["This is a sentence to embed"],
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
        try:
            if self.batching_enabled:
                return await self._batch_infer(inference_request)
            else:
                with torch.no_grad():
                    result = self.model.encode(
                        *inference_request.args, **inference_request.kwargs
                    )
                    return {"result": numpy_to_std(result)}
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

    @web_app.post("/reconfigure")
    def reconfigure(
        self, config: BatchingConfigUpdateRequest
    ) -> BatchingConfigUpdateResponse:

        message = []

        if config.max_batch_size is not None:
            self._batch_infer.set_max_batch_size(config.max_batch_size)
            message.append(f"max_batch_size updated to {config.max_batch_size}")

        if config.batch_wait_timeout_s is not None:
            self._batch_infer.set_batch_wait_timeout_s(config.batch_wait_timeout_s)
            message.append(
                f"batch_wait_timeout_s updated to {config.batch_wait_timeout_s}"
            )

        return BatchingConfigUpdateResponse(
            max_batch_size=self._batch_infer._get_max_batch_size(),
            batch_wait_timeout_s=self._batch_infer._get_batch_wait_timeout_s(),
            message=", ".join(message) if message else "No changes made",
        )

    @web_app.get("/batch_config")
    def get_batch_config(self) -> BatchingConfig:

        return BatchingConfig(
            max_batch_size=self._batch_infer._get_max_batch_size(),
            batch_wait_timeout_s=self._batch_infer._get_batch_wait_timeout_s(),
        )


def app_builder(args: dict) -> Application:
    return SentenceTransformersModelDeployment.bind(  # type: ignore[attr-defined]
        args["model_path"],
        args["trust_remote_code"],
        args["device"],
        args["torch_dtype"],
    )
