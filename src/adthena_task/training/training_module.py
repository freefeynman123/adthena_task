# flake8: noqa
"""Implements training class."""
from argparse import ArgumentParser

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from transformers import AdamW, get_linear_schedule_with_warmup

from adthena_task.config import Config
from adthena_task.models.bert import BertClassifier
from adthena_task.training.datamodule import weights
from adthena_task.training.utils import calculate_accuracy
from adthena_task.utils.metrics import F1Score

config = Config()


class BertLightningModule(pl.LightningModule):
    """Pytorch lightning based class which provides training logic."""

    def __init__(
        self,
        use_weights: bool = True,
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
        if use_weights:
            self.loss_function = nn.CrossEntropyLoss(weight=weights)
        else:
            self.loss_function = nn.CrossEntropyLoss()
        self.f1_score = F1Score()
        self.train_loss_epoch = []
        self.train_acc_epoch = []
        self.train_f1_epoch = []
        self.val_loss_epoch = []
        self.val_acc_epoch = []
        self.val_f1_epoch = []

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
        f1_score = self.f1_score(label_pred, labels)

        self.train_loss_epoch.append(loss.item())
        self.train_acc_epoch.append(acc.item())
        self.train_f1_epoch.append(f1_score.item())

        logs = {"train_loss": loss, "train_acc": acc, "train_f1": f1_score}
        return {"loss": loss, "train_acc": acc, "log": logs, "progress_bar": logs}

    def validation_step(self, batch, batch_idx):
        b_input_ids, b_attn_mask, labels = batch
        output = self(b_input_ids, b_attn_mask).squeeze()
        loss = self.loss_function(output, labels)
        label_pred = torch.argmax(output, dim=-1)
        acc = calculate_accuracy(label_pred, labels)
        f1_score = self.f1_score(label_pred, labels)
        self.log("val_loss", loss)
        self.val_loss_epoch.append(loss.item())
        self.val_acc_epoch.append(acc.item())
        self.val_f1_epoch.append(f1_score.item())
        logs = {"val_loss": loss, "val_acc": acc, "val_f1": f1_score}
        return {"loss": loss, "val_acc": acc, "log": logs, "progress_bar": logs}

    def on_epoch_end(self):
        metrics_to_log = {
            "train_loss_mean": np.mean(self.train_loss_epoch),
            "train_acc_mean": np.mean(self.train_acc_epoch),
            "train_f1_mean": np.mean(self.train_f1_epoch),
            "val_loss_mean": np.mean(self.train_loss_epoch),
            "val_acc_mean": np.mean(self.train_acc_epoch),
            "val_f1_mean": np.mean(self.train_f1_epoch),
        }
        self.logger.experiment.log_metrics(metrics_to_log, step=self.current_epoch)
        # reset for next epoch
        self.f1_score = F1Score()
        self.train_loss_epoch = []
        self.train_acc_epoch = []
        self.train_f1_epoch = []
        self.val_loss_epoch = []
        self.val_acc_epoch = []
        self.val_f1_epoch = []

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--use_weights", default=True, type=lambda x: str(x).lower() == "true"
        )
        parser.add_argument("--learning_rate", default=1e-5, type=float)
        parser.add_argument("--adam_epsilon", default=1e-8, type=float)
        parser.add_argument("--warmup_steps", default=0, type=int)
        parser.add_argument("--num_encoder_layers_to_train", default=2, type=int)
        return parser
