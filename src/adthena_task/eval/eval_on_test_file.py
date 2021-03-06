# flake8: noqa
from argparse import ArgumentParser
import datetime
import logging
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from adthena_task.config import Config
from adthena_task.eval.utils import create_model, device
from adthena_task.utils.timing import timed
from adthena_task.preprocessing.text_preprocessor import preprocessing_for_bert

config = Config()

base_path = os.path.dirname(__file__)

logging.basicConfig(
    handlers=[
        logging.FileHandler(
            filename=os.path.join(base_path, "log_records.txt"),
            encoding="utf-8",
            mode="a+",
        ),
        logging.StreamHandler(sys.stdout),
    ],
    format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
    datefmt="%F %A %T",
    level=logging.INFO,
)

current_time = f"{datetime.datetime.now():%Y%m%d%H%M}"


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--path_to_checkpoint", type=str, default=config.MODEL_PATH)
    parser.add_argument("--path_to_test_set", type=str, default=config.DATA_DIR_TEST)
    return parser.parse_known_args()


def eval(args: ArgumentParser):
    """Evaluation function on new data in txt format.
    Args:
        args: arguments for given run.

    Returns:
        None, saves results to csv file.

    """
    logging.info("Reading data")
    data = pd.read_table(args.path_to_test_set, header=None)
    data = data.iloc[:, 0]
    data = data.iloc[:3]
    ids, masks = timed(lambda: preprocessing_for_bert(data), logging)
    dataset = TensorDataset(ids.to(device), masks.to(device))
    test_loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    logging.info("Creating model.")
    model = timed(lambda: create_model(args.path_to_checkpoint), logging)
    predictions_list = []
    start = time.time()
    for batch_idx, (examples_ids, examples_masks) in enumerate(test_loader):
        with torch.no_grad():
            output = model(examples_ids, examples_masks)
            predictions = torch.argmax(output, dim=-1).cpu().detach().numpy()
            predictions_list.extend(predictions)
    elapsed = time.time() - start
    logging.info(f"Prediction took: {elapsed:0.3f} seconds")
    data = pd.DataFrame(data)
    data["predictions"] = np.array(predictions_list)
    logging.info("Writing results to csv.")
    data.to_csv(
        os.path.join(base_path, f"results_eval_{current_time}.csv"),
        header=None,
        index=None,
    )


if __name__ == "__main__":
    args, _ = parse_args()
    eval(args)
