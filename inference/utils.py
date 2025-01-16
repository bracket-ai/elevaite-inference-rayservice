import enum
from typing import Any, Dict, List

import numpy as np
import torch.cuda
from pydantic import BaseModel, Field, model_validator
from typing_extensions import Self


def numpy_to_std(obj):
    """Convert all objects in dict (recursively) from numpy types to vanilla
    Python types."""
    if isinstance(obj, list) or isinstance(obj, np.ndarray):
        new_obj = []
        for item in obj:
            new_obj.append(numpy_to_std(item))
        return new_obj
    elif isinstance(obj, dict):
        new_obj = {}
        for key, value in obj.items():
            new_obj[key] = numpy_to_std(value)
        return new_obj
    elif type(obj) in (int, float, str, bool) or obj is None:
        return obj
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, torch.dtype):
        return str(obj)
    else:
        return str(obj)


class InferenceRequest(BaseModel):
    args: List[Any] = Field(default=[])
    kwargs: Dict[str, Any] = Field(default={})


class BatchingConfigUpdateRequest(BaseModel):
    max_batch_size: int | None = Field(
        default=None,
        gt=0,
        description="Maximum number of requests to batch together",
    )
    batch_wait_timeout_s: float | None = Field(
        default=None,
        ge=0,
        description="Maximum time to wait for batch to fill up in seconds",
    )

    @model_validator(mode="before")
    def validate_at_least_one_not_null(self) -> Self:
        if self["max_batch_size"] is None and self["batch_wait_timeout_s"] is None:
            raise ValueError(
                "At least one of max_batch_size or batch_wait_timeout_s must be non-null"
            )
        return self


class BatchingConfig(BaseModel):
    max_batch_size: int | None
    batch_wait_timeout_s: float | None


class BatchingConfigUpdateResponse(BatchingConfig):
    message: str


def is_batchable_inference_request(request: InferenceRequest) -> bool:
    """
    Check if an inference request is batchable.

    A request is batchable if it has a single argument that is either a string,
    a dictionary, or a list of dictionaries.
    """

    # If a request has no args, it is not batchable
    # TODO: Support kwargs-only requests for sentence-transformers, which has only
    # 'sentences' as the argument for inputs to be encoded
    if not request.args:
        return False

    # If a request has more than one values in args, it is likely able to be batch processed
    # on its own, but cannot be processed with Elevaite's dynamic request batching, which
    # opportunistically constructs batches out of incoming requests.
    if len(request.args) != 1:
        return False

    arg = request.args[0]
    # Must be either a string, dict, or a list of dicts
    # list of dicts is a common pattern used for tasks like text-generation and document-qa
    # https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/text_generation.py#L264
    if isinstance(arg, (str, dict)):
        return True
    if isinstance(arg, list) and all(isinstance(item, dict) for item in arg):
        return True
    return False


class Library(enum.Enum):
    transformers = "transformers"
    sentence_transformers = "sentence-transformers"


dtype_mapping = {
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "bfloat16": torch.bfloat16,
    "complex64": torch.complex64,
    "complex128": torch.complex128,
    "uint8": torch.uint8,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "bool": torch.bool,
}
