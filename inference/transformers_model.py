import json
import logging
from http import HTTPStatus
from typing import Any, List

import torch.cuda
from fastapi import FastAPI, HTTPException
from ray import serve
from ray.serve import Application
from transformers import pipeline

from inference.utils import (
    BatchableInferenceRequest,
    BatchingConfig,
    BatchingConfigUpdateRequest,
    BatchingConfigUpdateResponse,
    dtype_mapping,
    numpy_to_std,
)

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


@serve.deployment(health_check_period_s=30, max_ongoing_requests=100)
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

        self._setup_batching()

    def _setup_batching(self):
        """Set up batching configuration based on model task and capabilities."""
        # Text generation will not work with batching unless the model has a pad token id
        if self.task != "text-generation":
            self.batching_enabled = True
            return

        self.batching_enabled = False
        # First check if the tokenizer has a pad token id
        if hasattr(self.pipe.tokenizer, "pad_token_id"):
            logger.info("Found pad_token_id attribute in tokenizer")
            if self.pipe.tokenizer.pad_token_id is not None:
                logger.info(
                    f"Tokenizer has pad token id: {self.pipe.tokenizer.pad_token_id}"
                )
                self.batching_enabled = True
            else:
                # Try to fall back to EOS token if available
                logger.info("Pad token id is None, checking for EOS token as fallback")
                if hasattr(self.pipe.model.config, "eos_token_id"):
                    eos_token_id = (
                        self.pipe.model.config.eos_token_id[0]
                        if isinstance(self.pipe.model.config.eos_token_id, list)
                        else self.pipe.model.config.eos_token_id
                    )
                    logger.info(f"Setting pad token id to eos token id: {eos_token_id}")
                    self.pipe.tokenizer.pad_token_id = eos_token_id
                    self.batching_enabled = True
                else:
                    logger.info(
                        "No EOS token ID found for fallback. Batching will not be supported."
                    )
        else:
            logger.info(
                "Tokenizer does not have pad_token_id attribute. Batching will not be supported."
            )

        logger.info(f"Batching enabled: {self.batching_enabled}")

    def _clear_cache(self):
        if str(self.pipe.device) == "cuda":
            torch.cuda.empty_cache()

    @web_app.get("/model_device")
    def model_device(self) -> str:
        return str(self.pipe.device)

    @serve.batch(max_batch_size=5, batch_wait_timeout_s=0.1)
    async def _batch_infer(
        self, requests: List[BatchableInferenceRequest]
    ) -> List[dict]:
        """
        Perform batch inference using the model pipeline.
        Batched inference for a set of requests requires that all requests have the same kwargs.
        This function groups requests by kwargs and performs parallel inference for each group serially.
        """

        try:
            logger.info(f"Starting batch call with {len(requests)} requests")
            results: List[dict] = []

            # group consecutive requests with the same kwargs
            current_group: List[Any] = []
            current_kwargs: str | None = None

            for request in requests:
                # Handle None kwargs by converting to empty dict
                kwargs_to_serialize = (
                    request.kwargs if request.kwargs is not None else {}
                )
                kwargs_key = json.dumps(kwargs_to_serialize, sort_keys=True)

                # If the kwargs have changed and we have a current group, perform inference
                if kwargs_key != current_kwargs and current_group:

                    if current_kwargs is None:
                        raise ValueError(
                            "current_kwargs should not be None at this point"
                        )

                    # perform inference for the current group
                    self._clear_cache()
                    logger.info(
                        f"Performing batch inference with batch size: {len(current_group)}"
                    )
                    with torch.no_grad():
                        group_results = self.pipe(
                            current_group,
                            **{
                                **json.loads(current_kwargs),
                                "batch_size": len(current_group),
                            },
                        )

                    if not isinstance(group_results, list):
                        group_results = [group_results]

                    logger.info(
                        f"Batch inference completed. Results length: {len(group_results)}"
                    )
                    results.extend({"result": numpy_to_std(r)} for r in group_results)

                    current_group = []

                # Args is only ever length 1
                # args[0] is only ever a list of dicts or a string, so we are safe to append it
                current_group.append(request.args[0])
                current_kwargs = kwargs_key

            # perform inference for the last group
            if current_group:
                if current_kwargs is None:
                    raise ValueError("current_kwargs should not be None at this point")

                self._clear_cache()
                logger.info(
                    f"Performing batch inference for the last group with batch size: {len(current_group)}"
                )
                with torch.no_grad():
                    group_results = self.pipe(
                        current_group,
                        **{
                            **json.loads(current_kwargs),
                            "batch_size": len(current_group),
                        },
                    )

                    if not isinstance(group_results, list):
                        group_results = [group_results]

                    logger.info(
                        f"Batch inference completed. Results length: {len(group_results)}"
                    )
                    results.extend({"result": numpy_to_std(r)} for r in group_results)

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
            "args": ["Help me write a poem that rhymes"],
            "kwargs": {"do_sample": false, "max_new_tokens": 50}
        }
        ```

        **Chat Example:**
        ```json
        {
            "args": [
                [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "What is the capital of France?"},
                    {"role": "assistant", "content": "The capital of France is Paris."},
                    {"role": "user", "content": "What is its population?"}
                ]
            ],
            "kwargs": {
                "do_sample": true,
                "temperature": 0.7,
                "max_new_tokens": 100
            }
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
        """
        if self.batching_enabled:
            try:
                return await self._batch_infer(inference_request)
            except Exception as e:
                logger.error(f"Batch Inference Error: {e}", exc_info=True)
                raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR)
            finally:
                self._clear_cache()
        else:
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

    @web_app.post("/reconfigure")
    def reconfigure(
        self, config: BatchingConfigUpdateRequest
    ) -> BatchingConfigUpdateResponse:
        """Update batch processing configuration."""
        if not self.batching_enabled:
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST,
                detail="Batching is not enabled for this model",
            )

        message = []

        if config.max_batch_size:
            self._batch_infer.set_max_batch_size(config.max_batch_size)
            message.append(f"max_batch_size updated to {config.max_batch_size}")

        if config.batch_wait_timeout_s:
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
        """Get current batch processing configuration."""

        if not self.batching_enabled:
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST,
                detail="Batching is not enabled for this model",
            )

        return BatchingConfig(
            max_batch_size=self._batch_infer._get_max_batch_size(),
            batch_wait_timeout_s=self._batch_infer._get_batch_wait_timeout_s(),
        )


def app_builder(args: dict) -> Application:
    return TransformersModelDeployment.bind(  # type: ignore[attr-defined]
        args["model_path"],
        args["task"],
        args["trust_remote_code"],
        args["device"],
        args["torch_dtype"],
    )
