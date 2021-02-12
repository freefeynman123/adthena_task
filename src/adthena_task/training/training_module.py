# flake8: noqa
"""Implements training class."""
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import torch.nn as nn
from transformers import AdamW, get_linear_schedule_with_warmup

from adthena_task.config import Config
from adthena_task.models.bert import BertClassifier
from adthena_task.training.utils import calculate_accuracy

config = Config()


class BertLightningModule(pl.LightningModule):
    """Pytorch lightning based class which provides training logic."""

    def __init__(
        self,
        learning_rate: float = 1e-5,
        adam_epsilon: float = 1e-8,
        warmup_steps: int = 0,
        min_epochs: int = 5,
        num_encoder_layers_to_train: int = 2,
        batch_size=config.BATCH_SIZE,
    ):
        super(BertLightningModule, self).__init__()
        self.save_hyperparameters()
        model = BertClassifier(
            config, num_encoder_layers_to_train=self.hparams.num_encoder_layers_to_train
        )
        self.model = model
        self.loss_function = nn.CrossEntropyLoss()

    def forward(self, *inputs):
        return self.model(*inputs)

    def configure_optimizers(self):
        optimizer = AdamW(
            self.model.parameters(),
            lr=1e-5,  # TODO: change to hparams after resolving lightning bug
            eps=self.hparams.adam_epsilon,
        )

        # Calculate total steps
        train_loader = self.train_dataloader()
        total_steps = (len(train_loader.dataset) // (self.hparams.batch_size)) // float(
            self.hparams.min_epochs
        )

        # Set up the learning rate scheduler
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            # Default value
            num_training_steps=total_steps,
        )
        return [optimizer], [scheduler]

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

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--learning_rate", default=1e-5, type=float)
        parser.add_argument("--adam_epsilon", default=1e-8, type=float)
        parser.add_argument("--warmup_steps", default=0, type=int)
        parser.add_argument("--num_encoder_layers_to_train", default=2, type=int)
        return parser
