import pytorch_lightning as pl
import torch
from pytorch_lightning import loggers
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, RichModelSummary
from pytorch_lightning.strategies import DDPStrategy

from libcore.dataset import build_dataset, build_dataloader, build_transform
from libcore.models import build_model
from .utils import ConfigSave


def train_net(cfg):
    data_cfg = cfg.DATA
    train_dataset, valid_dataset, cls_num = build_dataset(
        data_cfg.NAME.lower(), data_cfg.PATH, data_cfg.DATASET, build_transform(data_cfg.INPUT),
    )

    train_dataloader = build_dataloader(train_dataset, data_cfg.DATALOADER, shuffle=True)
    valid_dataloader = build_dataloader(valid_dataset, data_cfg.DATALOADER, shuffle=False)

    fgic_model = build_model(cfg.MODEL, cls_num)
    # checkpointer
    filename = cfg.MODEL.NAME \
               + '-' + data_cfg.NAME \
               + '-{epoch:03d}' \
               + '-{' + cfg.TRAINER.MONITOR_VALUE + ':03f}'

    model_ckpt = ModelCheckpoint(
        monitor=cfg.TRAINER.MONITOR_VALUE, mode='max',
        dirpath=cfg.SAVE_DIR, filename=filename,
        save_top_k=1, every_n_epochs=cfg.TRAINER.VALID_INTERVAL,
    )

    csv_logger = loggers.CSVLogger(save_dir=cfg.SAVE_DIR, version='csv')
    tes_logger = loggers.TensorBoardLogger(save_dir=cfg.SAVE_DIR, version='tensorboard')

    # build trainer
    callbacks = [
        model_ckpt,  # model_ckpt must be at the 0-th of the list
        LearningRateMonitor(),
        RichModelSummary(max_depth=5),
        ConfigSave(cfg.SAVE_DIR, cfg)
    ]
    trainer_cfg = cfg.TRAINER

    trainer = pl.Trainer(
        enable_progress_bar=False,
        logger=[csv_logger, tes_logger], max_epochs=trainer_cfg.MAX_EPOCHS,
        check_val_every_n_epoch=trainer_cfg.VALID_INTERVAL,
        sync_batchnorm=trainer_cfg.SYNC_BN,
        accelerator=trainer_cfg.DEVICE, devices=trainer_cfg.DEVICE_ID,

        precision='bf16-mixed',
        # gradient_clip_val=1.0,
        # gradient_clip_algorithm="value",

        strategy=DDPStrategy(find_unused_parameters=trainer_cfg.UNUSED_PARAMETERS),
        callbacks=callbacks, log_every_n_steps=30,

        num_nodes=1,
    )

    # start training
    trainer.fit(fgic_model, train_dataloader, valid_dataloader)
    trainer.validate(fgic_model, dataloaders=valid_dataloader)
