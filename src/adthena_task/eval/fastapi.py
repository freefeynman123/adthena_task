"""Minimal implementation of FastApi interface for model evaluation."""

from fastapi import FastAPI
import torch

from adthena_task.config import Config

config = Config()

app = FastAPI()

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def create_model() -> torch.nn.Module:
    """Creates model used for evaluation.

    Returns:
        Model for evaluation.
    """
    model_path = config.MODEL_PATH
    model = config.MODEL_EVAL(config)
    model.load_state_dict(torch.load(model_path, map_location=device))
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
    encoded = config.TOKENIZER.encode_plus(
        text,
        add_special_tokens=True,
        max_length=config.MAX_LEN,
        pad_to_max_length=True,
        return_attention_mask=True,
    )
    ids = torch.tensor(encoded["input_ids"], dtype=torch.long).unsqueeze(0)
    masks = torch.tensor(encoded["attention_mask"], dtype=torch.long).unsqueeze(0)
    ids = ids.to(device, dtype=torch.long)
    masks = masks.to(device, dtype=torch.long)
    with torch.no_grad():
        output = model(input_ids=ids, attention_mask=masks)
        return torch.argmax(output, dim=-1).cpu().detach().numpy()


@app.get("/predict/{sentence}")
def predict(sentence: str) -> dict:
    """Predicts label for provided sentence.

    Args:
        sentence: Provided sentence.

    Returns:
        Dict with label for provided sentence.
    """
    pred = prediction(sentence)[0]
    return {"Predicted label": str(pred)}
