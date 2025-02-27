# a demo code for fast load data for visualization
import re
import os
import torch

from pprint import pprint
from easydict import EasyDict as edict
from libcore.dataset import build_dataset, build_transform


def find_ckpt(ROOT_PATH, ckpt_dir):
    matcher = re.compile(r'best_acc=(0\.\d+)\.ckpt')

    best_acc = 0.0
    ckpt_name = None
    for file_name in os.listdir(os.path.join(ROOT_PATH, ckpt_dir)):
        print(file_name)
        if 'ckpt' in file_name:
            if file_name == 'last.ckpt':
                continue
            elif float(matcher.search(file_name).groups()[0]) > best_acc:
                ckpt_name = file_name

    if ckpt_name is None:
        raise 'No ckpt file find in %s' % os.listdir(os.path.join(ROOT_PATH, ckpt_dir))
    ckpt_path = os.path.join(ROOT_PATH, ckpt_dir, ckpt_name)

    return ckpt_path


def load_config(ROOT_PATH, ckpt_dir):
    import json
    config_file = os.path.join(ROOT_PATH, ckpt_dir, 'train_params.json')
    with open(config_file, 'r') as f:
        config = edict(json.load(f))
    print('loaded configure:')
    pprint(config)
    return config


def load_data(data_config, ):
    data_trans = build_transform(trans_name=data_config.trans_name, trans_params=data_config.trans_params)

    train_data, valid_data, cls_num = build_dataset(
        data_config.name, data_config.path, data_trans,
        split_train=False
    )
    return train_data, valid_data, cls_num


def load_net(network_builder, net_config, ckpt_path, cls_num):
    network = network_builder(net_config, cls_num)
    loaded_state_dict = torch.load(ckpt_path, map_location='cpu')['state_dict']
    new_state_dict = dict()
    for k in loaded_state_dict:
        new_state_dict[k[8:]] = loaded_state_dict[k]
    network.load_state_dict(new_state_dict, strict=False)
    return network
