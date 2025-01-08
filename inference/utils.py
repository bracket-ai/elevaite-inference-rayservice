import enum
from typing import Any, Dict, List

import numpy as np
import torch.cuda
from pydantic import BaseModel, Field, field_validator


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


class BatchingConfig(BaseModel):
    max_batch_size: int = Field(
        gt=0, description="Maximum number of requests to batch together"
    )
    batch_wait_timeout_s: float = Field(
        ge=0, description="Maximum time to wait for batch to fill up in seconds"
    )


class BatchableInferenceRequest(InferenceRequest):
    @field_validator("args", mode="before")
    def validate_args(cls, v):
        if not isinstance(v, list):
            raise ValueError("args must be a list")
        if len(v) != 1:
            raise ValueError(f"args must contain exactly one item (got {len(v)} items)")

        arg = v[0]
        # Must be either a string, dict, or a list of dicts
        # list of dicts is used for tasks like text-generation and document-qa
        # https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/text_generation.py#L264
        if isinstance(arg, (str, dict)):
            return v
        if isinstance(arg, list) and all(isinstance(item, dict) for item in arg):
            return v

        raise ValueError("args[0] must be either a string or a list of dictionaries")


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
