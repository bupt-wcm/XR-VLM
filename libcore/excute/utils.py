import datetime
import json
import logging
import os
import time
import string

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.rank_zero import rank_zero_only


@rank_zero_only
def console_log(save_dir):
    # configure logging at the root level of Lightning
    logging.getLogger("pytorch_lightning").setLevel(logging.INFO)

    # configure logging on module level, redirect to file
    logger = logging.getLogger("pytorch_lightning.core")
    logger.addHandler(logging.FileHandler(os.path.join(save_dir, 'train.log'), mode='w'))
    logger.addHandler(logging.StreamHandler())


def generate_unique_code(code_len=5):
    # generate the unique code and name for the exp
    code_str = string.ascii_letters + string.digits
    time_str = datetime.datetime.now().strftime("%Y%m%d%H%M")
    rng = np.random.default_rng(int(time.time()))
    dic_code = rng.choice(list(code_str), code_len, replace=True)
    exp_code = ''.join(dic_code) + '-' + time_str
    return exp_code


@rank_zero_only
def create_dir(config):
    os.mkdir(config.SAVE_DIR)


@rank_zero_only
def save_config(save_dir, config):
    with open(
            os.path.join(save_dir, 'train_params.json'), 'w'
    ) as f:
        json.dump(
            config,
            f,
            indent=4,
        )


class ConfigSave(Callback):
    def __init__(self, save_dir, cfg, ):
        self.save_dir = save_dir
        self.config = cfg

    def on_save_checkpoint(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", checkpoint):
        model_ckpt = trainer.checkpoint_callback
        config = self.config.clone()
        config.defrost()
        config['best_path'] = model_ckpt.best_model_path
        save_config(self.save_dir, config)
