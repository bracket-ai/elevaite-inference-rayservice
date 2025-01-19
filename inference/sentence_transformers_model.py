import itertools
import json
import logging
from http import HTTPStatus

import torch.cuda
from fastapi import FastAPI, HTTPException
from ray import serve
from ray.serve import Application
from sentence_transformers import SentenceTransformer

from inference.utils import (
    BatchingConfig,
    BatchingConfigUpdateRequest,
    BatchingConfigUpdateResponse,
    InferenceRequest,
    dtype_mapping,
    is_batchable_inference_request,
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
    async def _batch_infer(self, requests: list[InferenceRequest]) -> list[dict]:
        """
        SentenceTransformers models support batching, but only for a single set of additional
        encoding parameters per batch of input sentences. However, Ray's batch handler constructs
        batches of requests with no consideration for each request's contents. As an example,
        the following requests can be processed in a single batch:

        ```python
        requests = [
            {"args": ["This is a sentence to embed"], "kwargs": {"normalize_embeddings": True}},
            {"args": ["This is another sentence to embed"], "kwargs": {"normalize_embeddings": True}},
            {"args": ["This is a third sentence to embed"], "kwargs": {"normalize_embeddings": True}},
        ]
        ```

        -->

        batched_request = [
            {
                "args": [
                    "This is a sentence to embed",
                    "This is another sentence to embed",
                    "This is a third sentence to embed",
                ],
                "kwargs": {"normalize_embeddings": True}, #kwargs are the same for all requests
            }
        ]
        ```

        but the following set of requests cannot be processed in a single batch because they
        have different encoding params:

        ```python
        requests = [
            {"args": ["This is a sentence to embed"], "kwargs": {}},
            {"args": ["This is a sentence to embed"], "kwargs": {"normalize_embeddings": True}},
            {"args": ["This is a sentence to embed"], "kwargs": {"normalize_embeddings": False}},
        ]
        ```

        In order to allow batching while respecting the kwargs of each request, this function
        takes a Ray-compiled request batch (type: list[InferenceRequest]) and decomposes
        it into order-preserving "chunks" in which all requests have the same encoding params,
        It then performs parallel inference across requests within each sub-batch+kwargs pair.

        Example: Given a set of possible encoding kwargs {A, B} and a Ray-compiled batch of
        requests where both sets of kwargs are present, this function will group the Ray batch
        into four sub-batches:

            AAAABBAB -> 1: [AAAA], 2: [BB], 3: [A], 4: [B]

        and parallelize inference within each sub-batch. So instead of 8 calls, we only make 4.
        After inference, the results are returned in the same order as the requests were
        submitted.
        """
        logger.info(f"Starting batch call with {len(requests)} requests")
        results: list[dict] = []

        # group consecutive requests with the same kwargs
        for sub_batch_kwargs_key, sub_batch in itertools.groupby(
            requests,
            key=lambda request: json.dumps(request.kwargs or {}, sort_keys=True),
        ):
            sub_batch_args = [request.args[0] for request in list(sub_batch)]

            # perform inference for the current sub-batch
            logger.info(
                f"Performing batch inference with batch size: {len(sub_batch_args)}"
            )
            with torch.no_grad():
                sub_batch_results = self.model.encode(
                    sub_batch_args,
                    batch_size=len(sub_batch_args),
                    **json.loads(sub_batch_kwargs_key),
                )

            logger.info(
                f"Batch inference completed. Results shape: {sub_batch_results.shape}"
            )

            if len(sub_batch_results.shape) == 2:  # 2D array (batch, embedding_dim)
                results.extend(
                    {
                        "result": numpy_to_std(row),
                        "metadata": {
                            "batched": True,
                            "batch_size": len(sub_batch_args),
                        },
                        "warnings": [],
                    }
                    for row in sub_batch_results
                )
            else:  # Handle case where there's only one dimension
                results.append(
                    {
                        "result": numpy_to_std(sub_batch_results),
                        "metadata": {
                            "batched": True,
                            "batch_size": len(sub_batch_args),
                        },
                        "warnings": [],
                    }
                )

        assert len(results) == len(
            requests
        ), "Results length does not match requests length"

        return results

    @web_app.post("/infer")
    async def infer(self, inference_request: InferenceRequest) -> dict:
        """
        **Request Format:**

        The request must be formatted as a JSON object with two keys: `args` and `kwargs`.
        `args` is a list of strings, and `kwargs` is a dictionary of additional inference parameters.


        **Single Input:**
        ```json
        {
            "args": ["This is a sentence to embed"],
            "kwargs": {}
        }
        ```

        **Batch Input:**
        This is also not the most efficient way to batch inference. See 'note on batching'
        for more details.
        ```json
        {
            "args": [["This is a sentence to embed", "This is another sentence to embed"]],
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

        **A Note on Batching:**
        This model supports batching of requests sent in a single HTTP request (see above), but
        also allows for opportunistic batching of requests sent in separate HTTP requests to
        increase GPU utilization. "Batchable" requests should:
        - never use kwargs to store inputs to be encoded (i.e kwargs = {"sentences": ["This is a sentence to embed"]})
        - have a single input to be encoded (i.e args = ["This is a sentence to embed"]) so that Elevaite
        can opportunistically append it to a batch of other requests with the same encoding params.
        """

        # If batching is enabled and args is not empty, perform batch inference
        try:
            if (
                is_batchable_inference_request(inference_request)
                and self.batching_enabled
            ):
                return await self._batch_infer(inference_request)
            else:
                warnings = [
                    "Request uses a format which cannot be batched. For better performance, "
                    "use request format: {'args': [single_inference_input], 'kwargs': {other_params}}"
                ]

                with torch.no_grad():
                    result = self.model.encode(
                        *inference_request.args, **inference_request.kwargs
                    )

                    return {
                        "result": numpy_to_std(result),
                        "metadata": {
                            "batched": False,
                        },
                        "warnings": warnings,
                    }
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
