# flake8: noqa

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger

from adthena_task.config import Config
from adthena_task.training.datamodule import AdthenaDataModule
from adthena_task.training.trainer import Model

config = Config()

wandb_logger = WandbLogger(name="Adam-32-0.001", project="pytorchlightning")


def main():
    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=0.0, patience=5, verbose=True, mode="min"
    )
    model_checkpoint_callback = ModelCheckpoint(
        filepath="../results", monitor="val_loss", save_top_k=3, save_weights_only=True
    )

    trainer = pl.Trainer(
        gpus=0,
        callbacks=[early_stop_callback, model_checkpoint_callback],
        gradient_clip_val=config.GRADIENT_CLIP_VAL,
        min_epochs=10,
        logger=wandb_logger,
    )

    model = Model()
    dm = AdthenaDataModule()
    dm.setup()

    trainer.fit(model, dm)
    trainer.test()


if __name__ == "__main__":
    main()
