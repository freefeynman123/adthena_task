"""Preprocessing functions used for BERT model."""

import re
from typing import Union

import pandas as pd
import torch
from transformers import BertTokenizer

from adthena_task.config import Config

config = Config()


def text_preprocessing(text: str) -> str:
    """Preprocessing functions before inputting to BERT tokenizer.

    Args:
        text: Text to be preprocessed.

    Returns:
        Processed text.
    """
    text = re.sub(r"(@.*?)[\s]", " ", text)
    text = re.sub(r"[0-9]+", "", text)
    text = re.sub(r"\s([@][\w_-]+)", "", text).strip()
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace("#", " ")
    encoded_string = text.encode("ascii", "ignore")
    decode_string = encoded_string.decode()
    return decode_string


def preprocessing_for_bert(
    data: pd.DataFrame, tokenizer: BertTokenizer = config.TOKENIZER
) -> Union[torch.Tensor, torch.Tensor]:
    """Preprocessing function for BERT.

    Args:
        data: Data to be preprocessed in the form of pandas DataFrame.
        tokenizer: Tokenizer class used to encode sentences.

    Returns:
        Input ids and attention masks in the form of torch Tensor.

    """
    input_ids = []
    attention_masks = []

    for sent in data:
        encoded_sent = tokenizer.encode_plus(
            text=text_preprocessing(sent),
            add_special_tokens=True,
            max_length=config.MAX_LEN,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,  # Return attention mask
        )
        input_ids.append(encoded_sent.get("input_ids"))
        attention_masks.append(encoded_sent.get("attention_mask"))

    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    return input_ids, attention_masks
