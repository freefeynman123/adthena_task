"""File containing bert classifier used in the task."""

from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import BertModel


class BertClassifier(nn.Module):
    """Bert classifier with linear output."""

    def __init__(self, config: dataclass, num_encoder_layers_to_train: int = 2) -> None:
        """Init function for bert classifier.

        Raises:
            ValueError in case of num_encoder_layers_to_train higher
            than length of bert encoding module.

        Args:
            config: config file with parameters for given run
            num_encoder_layers_to_train: number of layers from bert encoder
                                        that are trained. By default we train 2 layers.
        """
        super(BertClassifier, self).__init__()
        D_in, H, D_out = (
            config.BERT_HIDDEN_SIZE,
            config.CLASSIFIER_HIDDEN_SIZE,
            config.N_LABEL,
        )

        self.bert = BertModel.from_pretrained("bert-base-uncased")

        # Instantiate an one-layer feed-forward classifier
        self.classifier = nn.Sequential(
            nn.Linear(D_in, H), nn.ReLU(), nn.Linear(H, D_out)
        )

        # Freeze the BERT model
        if num_encoder_layers_to_train:
            if num_encoder_layers_to_train > len(list(self.bert.encoder.layer)):
                raise ValueError(
                    "The number of layers to train is higher than number of "
                    "layers in bert encoder."
                )
            for param in self.bert.encoder.layer[:-1].parameters():
                param.requires_grad = False

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Forward method of BERT classifier. Returns logits for classification task.

        Args:
            input_ids: Ids from the BERT tokenizer.
            attention_mask: Attention mask from the BERT tokenizer.

        Returns:
            Logits for the classification task.
        """
        # Feed input to BERT
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Extract the last hidden state of the token `[CLS]` for classification task
        last_hidden_state = bert_outputs[0][:, 0, :]
        # Feed input to classifier to compute logits
        out = self.classifier(last_hidden_state)

        return out
