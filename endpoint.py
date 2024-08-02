import os
import subprocess
from typing import List

import numpy as np
import torch.cuda
from fastapi import FastAPI
from pydantic import BaseModel, Field
from ray import serve
from ray.serve import Application

from transformers import pipeline

app = FastAPI()


class CommandRequest(BaseModel):
    tokens: List[str] = []


@serve.deployment
@serve.ingress(app)
class ProbeDeployment:
    def __init__(self):
        pass

    @app.post("/run_command")
    def run_command(self, command: CommandRequest):
        result = subprocess.run(command.tokens, capture_output=True)
        return {
            "code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

def deployment(args) -> Application:
    return ProbeDeployment.bind()
