# flake8: noqa
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger
import torch

from adthena_task.config import Config
from adthena_task.training.datamodule import AdthenaDataModule
from adthena_task.training.training_module import BertLightningModule

config = Config()

wandb_logger = WandbLogger(name="Adam-32-0.001", project="pytorchlightning")


def parse_args(args=None):
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = AdthenaDataModule.add_argparse_args(parser)
    parser = BertLightningModule.add_model_specific_args(parser)
    parser.add_argument("--seed", type=int, default=config.SEED)
    return parser.parse_args(args)


def setup(args):
    pl.seed_everything(args.seed)
    dm = AdthenaDataModule.from_argparse_args(args)
    dm.setup()
    model = BertLightningModule()
    return dm, model


def main():
    n_gpus = torch.cuda.device_count()
    mocked_args = f"""
        --max_epochs 10
        --gpus {n_gpus}""".split()

    args = parse_args(mocked_args)

    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=0.0, patience=5, verbose=True, mode="min"
    )
    model_checkpoint_callback = ModelCheckpoint(
        dirpath="../eval",
        filename="{epoch}-{val_loss:.2f}-{other_metric:.2f}",
        monitor="val_loss",
        save_top_k=3,
        save_weights_only=True,
    )

    trainer = pl.Trainer(
        gpus=0,
        callbacks=[early_stop_callback, model_checkpoint_callback],
        gradient_clip_val=config.GRADIENT_CLIP_VAL,
        min_epochs=10,
    )

    dm, model = setup(args)

    trainer.fit(model, dm)
    trainer.test()


if __name__ == "__main__":
    main()
