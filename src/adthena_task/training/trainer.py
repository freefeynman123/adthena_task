# flake8: noqa
"""Implements training class."""

import pytorch_lightning as pl
import torch
import torch.nn as nn
from transformers import AdamW, get_linear_schedule_with_warmup

from adthena_task.config import Config
from adthena_task.models.bert import BertClassifier
from adthena_task.training.utils import calculate_accuracy

config = Config()


class Model(pl.LightningModule):
    """Pytorch lightning based class which provides training logic."""

    def __init__(self):
        super(Model, self).__init__()
        model = BertClassifier(config, freeze_bert=True)
        self.model = model
        self.loss_function = nn.CrossEntropyLoss()

    def forward(self, *inputs):
        return self.model(*inputs)

    def configure_optimizers(self):
        optimizer = AdamW(
            self.model.parameters(), lr=5e-5, eps=1e-8  # Default learning rate
        )
        # total_steps = len(self.train_dataloader()) * self.min_epochs

        # Set up the learning rate scheduler
        # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,  # Default value
        #     num_training_steps=total_steps)
        return optimizer

    def training_step(self, batch, batch_idx):
        b_input_ids, b_attn_mask, labels = batch

        output = self(b_input_ids, b_attn_mask)

        loss = self.loss_function(output, labels)
        label_pred = torch.argmax(output, dim=-1)
        acc = calculate_accuracy(label_pred, labels)

        self.log("train_loss", loss, on_step=False, on_epoch=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        b_input_ids, b_attn_mask, labels = batch
        output = self(b_input_ids, b_attn_mask).squeeze()
        loss = self.loss_function(output, labels)
        label_pred = torch.argmax(output, dim=-1)
        acc = calculate_accuracy(label_pred, labels)
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True)
        return output
