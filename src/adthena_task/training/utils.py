"""File containing utility function for model training."""
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch

from adthena_task.config import Config

config = Config()

torch.manual_seed(config.SEED)


def prepare_data() -> pd.DataFrame:
    """Prepares dataframe loaded from directory.

    Returns:
        Dataframe used in training.
    """
    data = pd.read_csv(config.DATA_DIR_TRAIN, header=None)
    data.columns = ["query", "label"]
    data["label_count"] = data.groupby("label")["label"].transform("count")
    return data


def prepare_weights(data: pd.DataFrame) -> np.ndarray:
    """Calculated optional weights for loss function.

    Used to lower the effect of class imbalance.

    Args:
        data: Input data.

    Returns:
        Numpy array with weights assigned to each class in ascending order.

    """
    data_label = data.loc[:, ["label", "label_count"]]
    data_temp = (
        data_label.groupby(by=["label", "label_count"], as_index=False)
        .first()
        .sort_values("label")
    )
    weights = 1 / data_temp.loc[:, "label_count"]
    weights = weights / max(weights)
    return weights.values


def get_train_val_split(
    data: pd.DataFrame, test_size: float = config.TEST_SIZE
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Prepares train test split for main dataframe.

    Args:
        data: Loaded dataframe.
        test_size: test size in split.

    Returns:
        Splitted dataset.
    """
    X = data.loc[:, "query"]
    y = data.loc[:, "label"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, random_state=config.SEED, test_size=test_size, stratify=y
    )
    return X_train, X_val, y_train, y_val


def calculate_accuracy(predict: torch.Tensor, y: torch.Tensor) -> float:
    """Calculates accuracy for given results and labels.

    Args:
        predict: Predicted values.
        y: Labels.

    Returns:
        Accuracy value.
    """
    true_predict = (predict == y).float()
    acc = true_predict.sum() / len(y)

    return acc
