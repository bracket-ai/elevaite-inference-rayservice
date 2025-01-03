import logging

import numpy as np
import torch
from colpali_engine import ColPali, ColPaliProcessor
from fastapi import FastAPI, File, Form, UploadFile
from PIL import Image
from pydantic import BaseModel
from ray import serve
from ray.serve import Application

from inference.utils import dtype_mapping

logger = logging.getLogger("ray.serve")


class InferenceRequest(BaseModel):
    queries: list[str]


web_app = FastAPI()


@serve.deployment(health_check_period_s=30, ray_actor_options={"num_gpus": 1})
@serve.ingress(web_app)
# NOTE: The name of this class must match the name of this application's
#       deployment on ElevAIte's RayService; i.e. don't rename it unless
#       you are sure you know what you're doing.
class ColPaliDeployment:
    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        torch_dtype: str = "float32",
    ):
        if device not in ["cuda", "cpu"]:
            raise ValueError("device must be one of 'cuda' or 'cpu'")

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is not available. Please check "
                "for available resources."
            )

        # Load the model into memory:
        self.model: ColPali = (
            ColPali.from_pretrained(
                pretrained_model_name_or_path=model_path,
                torch_dtype=dtype_mapping.get(torch_dtype.lower()),
                device_map=device,
            )
            .to(device)
            .eval()
        )
        print(f"Type of model: {type(self.model)}")
        self.processor: ColPaliProcessor = ColPaliProcessor.from_pretrained(model_path)
        print(f"Type of processor: {type(self.processor)}")

    @web_app.post("/embed_queries")
    def embed_queries(self, inference_request: InferenceRequest) -> list:
        queries = inference_request.queries
        batch_queries = self.processor.process_queries(queries).to(self.model.device)

        with torch.no_grad():
            query_embeddings_tensor = self.model(**batch_queries)
        query_embeddings: np.ndarray = query_embeddings_tensor.cpu().float().numpy()

        return query_embeddings.tolist()

    @web_app.post("/embed_images")
    def embed_images(self, image_files: list[UploadFile] = File([])):
        # Process the inputs
        images = [
            Image.open(image_file.file).convert("RGB") for image_file in image_files
        ]
        batch_images = self.processor.process_images(images).to(self.model.device)

        # Forward pass
        with torch.no_grad():
            image_embeddings_tensor = self.model(**batch_images)
        image_embeddings: np.ndarray = image_embeddings_tensor.cpu().float().numpy()

        return image_embeddings.tolist()

    @web_app.post("/score_queries_and_images")
    def score_queries_and_images(
        self, queries: list[str] = Form(), image_files: list[UploadFile] = File([])
    ):
        """
        Example call:

        ```
        files = [('image_files', (img_path, open(img_path, 'rb').read(), 'image/jpeg')) for img_path in ['santa.jpg', 'easter_bunny.jpg']]
        response = requests.post("http://localhost:8000/score_queries_and_images", files=files, data={"queries": ["Santa Claus", "Tooth Fairy"]})
        response.json()
        >>> [[13.647987365722656, 9.606452941894531], [7.7808074951171875, 9.359827041625977]]
        ```
        """
        images = [
            Image.open(image_file.file).convert("RGB") for image_file in image_files
        ]

        print(f"{images=}")
        print(f"{queries=}")

        # Process the inputs
        batch_images = self.processor.process_images(images).to(self.model.device)
        batch_queries = self.processor.process_queries(queries).to(self.model.device)

        # Forward pass
        with torch.no_grad():
            image_embeddings = self.model(**batch_images)
            query_embeddings = self.model(**batch_queries)

        scores = self.processor.score_multi_vector(query_embeddings, image_embeddings)
        return scores.cpu().numpy().tolist()


def app_builder(args: dict) -> Application:
    return ColPaliDeployment.bind(  # type: ignore[attr-defined]
        args["model_path"],
        args["device"],
    )
