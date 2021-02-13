"""Minimal implementation of FastApi interface for model evaluation."""

from fastapi import FastAPI
import torch

from adthena_task.config import Config
from adthena_task.eval.utils import create_model, prediction

config = Config()

app = FastAPI()

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


@app.get("/predict/{sentence}")
def predict(sentence: str) -> dict:
    """Predicts label for provided sentence.

    Args:
        sentence: Provided sentence.

    Returns:
        Dict with label for provided sentence.
    """
    model = create_model()
    pred = prediction(sentence, model)[0]
    return {"Predicted label": str(pred)}
