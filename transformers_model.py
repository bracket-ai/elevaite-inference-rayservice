import gc
import logging
from http import HTTPStatus

import torch.cuda
from fastapi import FastAPI, HTTPException
from ray import serve
from ray.serve import Application
from transformers import pipeline

from utils import InferenceRequest, dtype_mapping, numpy_to_std

web_app = FastAPI()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ray")


@serve.deployment()
@serve.ingress(web_app)
class TransformersModelDeployment:
    def __init__(
        self,
        model_path: str,
        task: str,
        trust_remote_code: bool,
        device: str = "auto",
        torch_dtype: str | None = None,
    ):
        logger.info(
            f"Initializing TransformersModelDeployment with model_path={model_path}, task={task}, device={device}"
        )
        self.model_path = model_path
        self.task = task
        self.trust_remote_code = trust_remote_code
        self.torch_dtype = torch_dtype

        if device not in ["cuda", "auto", "cpu"]:
            logger.error(f"Invalid device specified: {device}")
            raise ValueError("device must be one of 'auto', 'cuda', or 'cpu'")

        if device == "cuda" and not torch.cuda.is_available():
            logger.error("CUDA was requested but is not available")
            raise RuntimeError(
                "CUDA was requested but is not available. Please check "
                "for available resources."
            )

        pipe_kwargs = {
            "task": self.task,
            "model": self.model_path,
            "trust_remote_code": self.trust_remote_code,
        }

        if device == "cuda":
            pipe_kwargs["device"] = 0  # Explicitly set to GPU index 0
        elif device == "auto":
            pipe_kwargs["device_map"] = "auto"
        else:
            pipe_kwargs["device"] = "cpu"

        if torch_dtype:
            pipe_kwargs["torch_dtype"] = dtype_mapping.get(torch_dtype.lower(), None)

        logger.info(f"Initializing pipeline with kwargs: {pipe_kwargs}")
        # Initialize the pipeline with device_map="auto"
        self.pipe = pipeline(**pipe_kwargs)

        # Resize token embeddings to match the tokenizer
        initial_embedding_size = self.pipe.model.get_input_embeddings().num_embeddings
        self.pipe.model.resize_token_embeddings(len(self.pipe.tokenizer))
        new_embedding_size = self.pipe.model.get_input_embeddings().num_embeddings
        # Log the change in embedding size
        logger.info(
            f"Token embeddings size: initial={initial_embedding_size}, new={new_embedding_size}"
        )

        self.pipe.model.eval()
        logger.info("Model initialization complete")

    def _clear_cache(self):
        logger.info("Clearing cache")
        if torch.cuda.is_available():
            logger.info(
                f"GPU memory allocated before cache clear: {torch.cuda.memory_allocated() / 1e9} GB"
            )
            logger.info(
                f"GPU memory reserved before cache clear: {torch.cuda.memory_reserved() / 1e9} GB"
            )
            torch.cuda.empty_cache()
            logger.info(
                f"GPU memory allocated after cache clear: {torch.cuda.memory_allocated() / 1e9} GB"
            )
            logger.info(
                f"GPU memory reserved after cache clear: {torch.cuda.memory_reserved() / 1e9} GB"
            )
        gc.collect()

    @web_app.get("/model_device")
    def model_device(self) -> str:
        device = str(next(self.pipe.model.parameters()).device)
        logger.info(f"Model device: {device}")
        return device

    @web_app.get("/model_device_map")
    def model_device_map(self) -> dict:
        logger.info("Retrieving model device map")
        if hasattr(self.pipe.model, "hf_device_map"):
            return self.pipe.model.hf_device_map
        else:
            logger.warning("Model does not have hf_device_map attribute")
            return {"error": "Model does not have hf_device_map attribute"}

    def _memory_summary(self):
        logger.info("Retrieving CUDA memory summary")
        if torch.cuda.is_available():
            return torch.cuda.memory_summary()
        else:
            return "CUDA is not available on this system."

    @web_app.post("/infer")
    def infer(self, inference_request: InferenceRequest) -> dict:
        logger.info("Received inference request")
        args = inference_request.args
        kwargs = inference_request.kwargs

        try:
            self._clear_cache()
            if torch.cuda.is_available():
                logger.info(
                    f"GPU memory allocated before inference: {torch.cuda.memory_allocated() / 1e9} GB"
                )
                logger.info(
                    f"GPU memory reserved before inference: {torch.cuda.memory_reserved() / 1e9} GB"
                )
            with torch.no_grad():
                logger.info("Running inference")
                result = self.pipe(*args, **kwargs)
                logger.info(f"Inference result: {result}")
            if torch.cuda.is_available():
                logger.info(
                    f"GPU memory allocated after inference: {torch.cuda.memory_allocated() / 1e9} GB"
                )
                logger.info(
                    f"GPU memory reserved after inference: {torch.cuda.memory_reserved() / 1e9} GB"
                )

            out = numpy_to_std(result)
            del result, args, kwargs  # Delete variables
            self._clear_cache()
            logger.info("Inference completed successfully")
            return {"result": out}
        except torch.cuda.OutOfMemoryError as oom_error:
            logger.error(f"CUDA out of memory error: {str(oom_error)}", exc_info=True)
            logger.error(f"Memory summary:\n{self._memory_summary()}")
            self._clear_cache()
            raise HTTPException(
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                detail=f"CUDA out of memory error: {str(oom_error)}",
            )
        except Exception as e:
            logger.error(f"Error during inference: {str(e)}", exc_info=True)
            self._clear_cache()
            raise HTTPException(
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail=str(e)
            )

    @web_app.get("/model_config")
    def model_config(self):
        logger.info("Retrieving model config")
        return numpy_to_std(self.pipe.model.config.__dict__)

    @web_app.get("/compute_config")
    def get_num_threads(self):
        import os

        logger.info("Retrieving thread information")
        return {
            "CPU_COUNT": os.cpu_count(),
            "NUM_THREADS": torch.get_num_threads(),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", None),
            "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", None),
            "CUDA_LAUNCH_BLOCKING": os.environ.get("CUDA_LAUNCH_BLOCKING", None),
        }

    @web_app.get("/memory_summary")
    def memory_summary(self):
        return self._memory_summary()


def app_builder(args: dict) -> Application:
    logger.info(f"Building application with args: {args}")
    return TransformersModelDeployment.bind(  # type: ignore[attr-defined]
        args["model_path"],
        args["task"],
        args["trust_remote_code"],
        args["device"],
        args["torch_dtype"],
    )
