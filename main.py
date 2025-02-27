import os
import torch
import models
import argparse

from pprint import pprint
from libcore.excute import train_net, valid_net
from libcore.excute.utils import generate_unique_code, create_dir
from libcore.config import get_cfg


def set_up(args):
    # Initialize the main config and updater
    cfg, updater = get_cfg()

    # Load data-specific configuration safely
    data_cfg, _ = get_cfg(args.data)
    cfg.merge_from_other_cfg(data_cfg)

    # Merge from config file if provided
    if args.config:
        cfg.merge_from_file(args.config)  # Fixed indentation here

    # Merge from command line options
    if args.opts:
        cfg.merge_from_list(args.opts)

    # Create output directory with safe path handling
    exp_code = generate_unique_code()
    save_dir = os.path.join('./ckpts', f"{exp_code}-{args.data.upper()}-{cfg.DESCRIBE}".replace(' ', '_'))
    cfg.EXP_CODE = exp_code
    cfg.SAVE_DIR = save_dir
    cfg = updater.update(cfg)  # Apply updater function properly
    # Finalize config and verify directory creation
    cfg.freeze()  # Prevent further modification
    create_dir(cfg)  # Ensure directory creation after setting SAVE_DIR

    print("Final configuration:")
    pprint(cfg)
    return cfg


if __name__ == '__main__':
    # torch.autograd.set_detect_anomaly(True)
    torch.set_float32_matmul_precision('highest')
    parser = argparse.ArgumentParser('MppCLIP')
    parser.add_argument('--data', type=str)
    parser.add_argument('--config', type=str, metavar='FILE')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--seed', type=int, default=632)
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    # torch.set_float32_matmul_precision('highest')
    # pl.seed_everything(args.seed)
    if args.evaluate:
        valid_net(args)  # validate the learned network without DDP
        exit(0)

    exp_cfg = set_up(args)
    train_net(exp_cfg)
