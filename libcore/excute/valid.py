import json
from libcore.config import get_cfg
from pprint import pprint
import pytorch_lightning as pl
import torch.cuda
from easydict import EasyDict as edict

from libcore.dataset import build_dataset, build_dataloader, build_transform
from libcore.models import build_model


def valid_net(args):
    cfg, updater = get_cfg()
    cfg.merge_from_file(args.config)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    pprint(cfg)

    torch.cuda.empty_cache()
    data_cfg = cfg.DATA
    _, valid_dataset, cls_num = build_dataset(
        data_cfg.NAME.lower(), data_cfg.PATH, data_cfg.DATASET, build_transform(data_cfg.INPUT),
    )
    valid_dataloader = build_dataloader(valid_dataset, data_cfg.DATALOADER, shuffle=False)

    fgic_model = build_model(cfg.MODEL, cls_num)
    fgic_model.load_state_dict(
        torch.load(cfg['best_path'], map_location='cpu')['state_dict'], strict=False,
    )

    # build network
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=cfg.TRAINER.DEVICE_ID,
        num_nodes=1,
        max_epochs=1,
    )
    trainer.validate(fgic_model, valid_dataloader, verbose=True)
