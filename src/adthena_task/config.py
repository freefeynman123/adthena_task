"""Config file."""

from dataclasses import dataclass
import os

from transformers import BertTokenizer

base_path = os.path.dirname(__file__)


@dataclass
class Config:
    """Config file for training and evaluating."""

    # Data location params
    DATA_DIR_TRAIN: str = os.path.join(base_path, "data", "trainSet.csv")
    DATA_DIR_TEST: str = os.path.join(base_path, "data", "candidateTestSet.txt")

    # Model params
    TOKENIZER: BertTokenizer = BertTokenizer.from_pretrained(
        "bert-base-uncased", do_lower_case=True
    )
    N_LABEL: int = 1419
    BERT_HIDDEN_SIZE: int = 768
    CLASSIFIER_HIDDEN_SIZE: int = 50
    MAX_LEN: int = 30
    MODEL_PATH: str = "checkpoints/epoch=0-val_loss=7.25-other_metric=0.00.ckpt"

    # Training params
    BATCH_SIZE: int = 32
    GRADIENT_CLIP_VAL: float = 1.0
    SEED: int = 42
    TEST_SIZE: float = 0.5
