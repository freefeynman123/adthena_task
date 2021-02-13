"""File with utilities for prediction."""

import torch

from adthena_task.config import Config
from adthena_task.preprocessing.text_preprocessor import preprocessing_for_bert
from adthena_task.training.training_module import BertLightningModule

config = Config()

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def create_model(path: str = config.MODEL_PATH) -> torch.nn.Module:
    """Creates model used for evaluation.

    Args:
        path: Path to model.

    Returns:
        Model for evaluation.
    """
    model = BertLightningModule.load_from_checkpoint(path)
    model.to(device)
    model.eval()
    return model


def prediction(text: str, model: torch.nn.Module) -> str:
    """Prediction function for new input.

    Args:
        text: Text to be inserted.
        model: Model

    Returns:
        Predicted class label.
    """
    ids, masks = preprocessing_for_bert(text, config.TOKENIZER)
    with torch.no_grad():
        output = model(ids, masks)
        return torch.argmax(output, dim=-1).cpu().detach().numpy()
