"""Config file."""

import os

from transformers import BertTokenizer

from adthena_task.models.bert import BertClassifier

base_path = os.path.dirname(__file__)

DATA_DIR_TRAIN = os.path.join(base_path, "data", "train_600_plus.csv")
DATA_DIR_TEST = os.path.join(base_path, "data", "candidateTestSet.txt")

TOKENIZER = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
N_LABEL = 1419
BERT_HIDDEN_SIZE = 768
CLASSIFIER_HIDDEN_SIZE = 50
MAX_LEN = 30
MODEL_EVAL = BertClassifier
MODEL_PATH = "epoch_09.pt"

BATCH_SIZE = 128
GRADIENT_CLIP_VAL = 1.0
SEED = 42
TEST_SIZE = 0.5
