# flake8: noqa
from argparse import ArgumentParser

import pandas as pd
from torch.utils.data import DataLoader

from adthena_task.config import Config
from adthena_task.eval.utils import create_model, prediction

config = Config()


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--path_to_checkpoint", type=str, default=config.MODEL_PATH)
    parser.add_argument("--path_to_test_set", type=str, default=config.DATA_DIR_TEST)
    return parser.parse_args()


def eval(args: ArgumentParser):
    """Evaluation function on new data in txt format.
    Args:
        args:

    Returns:

    """
    data = pd.read_table(args.path_to_test_set).values
    test_loader = DataLoader(data, batch_size=config.BATCH_SIZE, shuffle=False)
    model = create_model(args.path_to_checkpoint)
    predictions_list = []
    for examples in test_loader:
        predictions_list.append(prediction(examples, model))
    return predictions_list


if __name__ == "__main__":
    args = parse_args()
    eval(args)
