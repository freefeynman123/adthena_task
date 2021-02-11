"""Implements data loading class for adthena task."""

from pytorch_lightning.core.datamodule import LightningDataModule
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

from adthena_task.config import Config
from adthena_task.preprocessing.text_preprocessor import preprocessing_for_bert
from adthena_task.training.utils import get_train_val_split, prepare_data

config = Config()


class AdthenaDataModule(LightningDataModule):
    """Implements data loading utilities for query search problem."""

    def setup(self) -> None:
        """Setup for training and validation data."""
        data = prepare_data()
        X_train, X_val, y_train, y_val = get_train_val_split(data)
        X_val, X_test, y_val, y_test = train_test_split(
            X_val, y_val, random_state=config.SEED, test_size=0.5, stratify=y_val
        )
        train_inputs, train_masks = preprocessing_for_bert(X_train)
        val_inputs, val_masks = preprocessing_for_bert(X_val)
        test_inputs, test_masks = preprocessing_for_bert(X_test)
        train_labels = torch.tensor(y_train.values)
        val_labels = torch.tensor(y_val.values)
        test_labels = torch.tensor(y_test.values)
        self.train_data = TensorDataset(train_inputs, train_masks, train_labels)
        self.val_data = TensorDataset(val_inputs, val_masks, val_labels)
        self.test_data = TensorDataset(test_inputs, test_masks, test_labels)

    def train_dataloader(self) -> DataLoader:
        """Prepares train dataloader.

        Returns:
            Train dataloader.
        """
        train_sampler = RandomSampler(self.train_data)
        train_dataloader = DataLoader(
            self.train_data, sampler=train_sampler, batch_size=config.BATCH_SIZE
        )
        return train_dataloader

    def val_dataloader(self) -> DataLoader:
        """Prepares validation dataloader.

        Returns:
            Validation dataloader.
        """
        val_sampler = SequentialSampler(self.val_data)
        val_dataloader = DataLoader(
            self.val_data, sampler=val_sampler, batch_size=config.BATCH_SIZE
        )
        return val_dataloader

    def test_dataloader(self) -> DataLoader:
        """Prepares test dataloader.

        Returns:
            Test dataloader.
        """
        test_sampler = SequentialSampler(self.test_data)
        test_dataloader = DataLoader(
            self.test_data, sampler=test_sampler, batch_size=config.BATCH_SIZE
        )
        return test_dataloader
