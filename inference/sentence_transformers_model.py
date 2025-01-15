import json
import logging
from http import HTTPStatus
from typing import Any

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
        self, requests: list[BatchableInferenceRequest]
    ) -> list[dict]:
        """
        SentenceTransformers models support batching, but only for a single set of kwargs per
        batch of inputs. However, Ray's own batch handler constructs batches of requests with no
        consideration for each request's contents. In order to allow batching while
        respecting the kwargs of each request, this function takes a Ray-compiled batch of
        requests and decomposes them into sub-batches where all requests have the same kwargs,
        while preserving the order of the requests. It then performs parallel inference for
        each sub-batch-kwargs pair, one at a time.

        Example: Given a set of possible kwargs {A, B} and a Ray-compiled batch of
        requests where both sets of kwargs are present, this function will group the Ray batch
        into four sub-batches:

            AAAABBAB -> 1: [AAAA], 2: [BB], 3: [A], 4: [B]

        and parallelize inference within each sub-batch. So instead of 8 calls, we only make 4.
        After inference, the results are returned in the same order as the requests were
        submitted.
        """
        try:
            logger.info(f"Starting batch call with {len(requests)} requests")
            results: list[dict] = []

            # Group by kwargs since encode() supports different options
            current_sub_batch_args: list[Any] = []
            current_sub_batch_kwargs_key: str = "{}"

            for request in requests:
                kwargs_to_serialize = request.kwargs or {}
                kwargs_key = json.dumps(kwargs_to_serialize, sort_keys=True)

                # If the kwargs have changed and we have a current group, perform inference
                if (
                    kwargs_key != current_sub_batch_kwargs_key
                    and current_sub_batch_args
                ):

                    # Process current group
                    with torch.no_grad():
                        # returns a numpy.ndarray of shape (n_sentences, embedding_dim)
                        sub_batch_results = self.model.encode(
                            current_sub_batch_args,
                            batch_size=len(current_sub_batch_args),
                            **json.loads(current_sub_batch_kwargs_key),
                        )

                    if (
                        len(sub_batch_results.shape) == 2
                    ):  # 2D array (batch, embedding_dim)
                        results.extend(
                            {"result": numpy_to_std(row)} for row in sub_batch_results
                        )
                    else:  # Handle case where there's only one dimension (single embedding)
                        results.append({"result": numpy_to_std(sub_batch_results)})
                    current_sub_batch_args = []

                # args[0] is guaranteed to be string, dict, or list of dicts
                current_sub_batch_args.append(request.args[0])
                current_sub_batch_kwargs_key = kwargs_key

            # Process final group
            if current_sub_batch_args:

                with torch.no_grad():
                    sub_batch_results = self.model.encode(
                        current_sub_batch_args,
                        batch_size=len(current_sub_batch_args),
                        **json.loads(current_sub_batch_kwargs_key),
                    )
                if len(sub_batch_results.shape) == 2:  # 2D array (batch, embedding_dim)
                    results.extend(
                        {"result": numpy_to_std(row)} for row in sub_batch_results
                    )
                else:  # Handle case where there's only one dimension
                    results.append({"result": numpy_to_std(sub_batch_results)})

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
