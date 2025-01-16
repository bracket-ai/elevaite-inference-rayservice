import json
import logging
from http import HTTPStatus
from typing import Any

import torch.cuda
from fastapi import FastAPI, HTTPException
from ray import serve
from ray.serve import Application
from transformers import pipeline

from inference.utils import (
    BatchingConfig,
    BatchingConfigUpdateRequest,
    BatchingConfigUpdateResponse,
    InferenceRequest,
    dtype_mapping,
    is_batchable_inference_request,
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
    async def _batch_infer(self, requests: list[InferenceRequest]) -> list[dict]:
        """
        Transformers pipelines support batching, but only for a single set of adttional
        inference parameters per batch of text inputs. However, Ray's batching mechanism constructs
        batches of requests with no consideration for each request's contents. As an example,
        the following requests can be processed in a single batch:

        ```python
        requests = [
            {"args": ["Hello world!"], "kwargs": {"do_sample": True}},
            {"args": ["Is anyone there?"], "kwargs": {"do_sample": True}},
            {"args": ["Can you hear me?"], "kwargs": {"do_sample": True}},
        ]

        -->

        batched_request = [
            {
                "args": [
                    "Hello world!",
                    "Is anyone there!",
                    "Can you hear me!"
                ],
                "kwargs": {"do_sample": True} #kwargs are the same for all requests
            }
        ]
        ```


        but the following set of requests cannot be processed in a single batch because they
        have different inference params:

        ```python
        requests = [
            {"args": ["Hello world!"], "kwargs": {}},
            {"args": ["Is anyone there?"], "kwargs": {"do_sample": True}},
            {"args": ["Can you hear me?"], "kwargs": {"do_sample": False}},
        ]
        ```

        In order to allow batched inference while respecting the params of each request, this
        function takes a Ray-compiled batch of requests and decomposes it into order-preserving
        "chunks" in which all requests have the same inference params. It then performs parallel
        inference across requests within each sub-batch+params pair.

        Example: Given a set of possible kwargs {A, B} and a Ray-compiled batch of
        requests where both sets of kwargs are present, this function will group the Ray batch
        into the following sub-batches:

            [AAAABBAB] -> 1: [AAAA], 2: [BB], 3: [A], 4: [B]

        and parallelize inference within each sub-batch. So instead of 8 calls, we only make 4.
        After inference, the results are returned in the same order as the requests were
        submitted.
        """

        logger.info(f"Starting batch call with {len(requests)} requests")
        results: list[dict] = []

        # group consecutive requests with the same kwargs
        current_sub_batch_args: list[Any] = []
        current_sub_batch_kwargs_key: str = "{}"

        for request in requests:
            kwargs_to_serialize = request.kwargs or {}
            kwargs_key = json.dumps(kwargs_to_serialize, sort_keys=True)

            # If the kwargs have changed and we have a current sub-batch, perform inference
            if kwargs_key != current_sub_batch_kwargs_key and current_sub_batch_args:

                # perform inference for the current sub-batch
                self._clear_cache()
                logger.info(
                    f"Performing batch inference with batch size: {len(current_sub_batch_args)}"
                )
                with torch.no_grad():
                    sub_batch_results = self.pipe(
                        current_sub_batch_args,
                        **{
                            **json.loads(current_sub_batch_kwargs_key),
                            "batch_size": len(current_sub_batch_args),
                        },
                    )

                if not isinstance(sub_batch_results, list):
                    sub_batch_results = [sub_batch_results]

                logger.info(
                    f"Batch inference completed. Results length: {len(sub_batch_results)}"
                )
                results.extend({"result": numpy_to_std(r)} for r in sub_batch_results)

                current_sub_batch_args = []

            # Args is only ever length 1
            # args[0] is only ever a list of dicts or a string, so we are safe to append it
            current_sub_batch_args.append(request.args[0])
            current_sub_batch_kwargs_key = kwargs_key

        # perform inference for the last sub-batch
        if current_sub_batch_args:

            self._clear_cache()
            logger.info(
                f"Performing batch inference for the last sub-batch with batch size: {len(current_sub_batch_args)}"
            )
            with torch.no_grad():
                sub_batch_results = self.pipe(
                    current_sub_batch_args,
                    **{
                        **json.loads(current_sub_batch_kwargs_key),
                        "batch_size": len(current_sub_batch_args),
                    },
                )

                if not isinstance(sub_batch_results, list):
                    sub_batch_results = [sub_batch_results]

                logger.info(
                    f"Batch inference completed. Results length: {len(sub_batch_results)}"
                )
                results.extend({"result": numpy_to_std(r)} for r in sub_batch_results)

        return results

    @web_app.post("/infer")
    async def infer(self, inference_request: InferenceRequest) -> dict:
        """
        **Request Format:**

        ***Single Text Generation:***
        ```json
        {
            "args": ["Help me write a poem that rhymes"],
            "kwargs": {"do_sample": false, "max_new_tokens": 50}
        }
        ```

        ***Batched Text Generation:***
        Note: The `batch_size` kwarg is required for batched inference on transformers models.
        This is also not the most efficient way to batch inference. See 'note on batching'
        for more details.
        ```json
        [
            {
                "args": [
                    "Help me write a poem that rhymes",
                    "Help me write a haiku that rhymes. Good luck.",
                    "Help me write a limerick with 6 lines."
                ],
                "kwargs": {
                    "do_sample": false,
                    "max_new_tokens": 50,
                    "batch_size": 3
                }
            }
        ]
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

        **A Note on Batching:**
        This model supports batching of requests sent in a single HTTP request (see above), but
        also allows for opportunistic batching of requests sent in separate HTTP requests to
        increase GPU utilization. "Batchable" requests should:
        - never use kwargs to store inference inputs (i.e kwargs = {"text_inputs": ["Write me a poem"]})
        - have a single input in args (i.e args = ["Generate me a docstring for this code"]) so that Elevaite
        can opportunistically append it to a batch of other requests with the same inference params.
        """

        # If batching is enabled and args is not empty, perform batch inference
        if is_batchable_inference_request(inference_request) and self.batching_enabled:
            try:
                return await self._batch_infer(inference_request)
            except Exception as e:
                logger.error(f"Batch Inference Error: {e}", exc_info=True)
                raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR)
            finally:
                self._clear_cache()
        else:
            warnings = [
                "Request uses a format which cannot be batched. For better performance, "
                "use request format: {'args': [single_inference_input], 'kwargs': {other_params}}"
            ]

            args = inference_request.args
            kwargs = inference_request.kwargs

            try:
                self._clear_cache()
                with torch.no_grad():
                    result = self.pipe(*args, **kwargs)
                return {"result": numpy_to_std(result), "warnings": warnings}
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
