import os.path
import pickle
import random
from typing import Tuple
from collections import defaultdict

import numpy as np
import sklearn
from PIL import Image
from tabulate import tabulate
from torch.utils.data import Dataset, DataLoader

from libcore.dataset.data_loaders import load_func_dict

__all__ = ['build_dataset', 'build_dataloader']
CACHE_PATH = './cached_data'


def read_im(im_path):
    with Image.open(im_path).convert('RGB') as im:
        # this size is bigger than the size in transform functions, little effect for performance
        # im = torchtfs.Resize(size=600)(im)
        return im


class FineGrainedDataset(Dataset):
    def __init__(self, data_list, im_transforms, use_cache=False):
        self.data_list = data_list
        self.use_cache = use_cache
        self.im_transforms = im_transforms

    def __getitem__(self, index):
        im_path, label = self.data_list[index]
        if not self.use_cache:
            im = read_im(im_path)
            im = self.im_transforms(im)
        else:
            im = self.im_transforms(im_path)
        return im, int(label)

    def __len__(self):
        return len(self.data_list)


def build_dataset(data_name, data_path, dataset_cfg, trans_func) -> Tuple[FineGrainedDataset, FineGrainedDataset, int]:

    if dataset_cfg.CACHED:
        cache_file_name = '{:s}_cache.pkl'.format(data_name)
        cache_file_path = os.path.join(CACHE_PATH, cache_file_name)
        if os.path.exists(cache_file_path):
            with open(cache_file_path, 'rb') as f:
                train_data, valid_data = pickle.load(f)
        else:
            data_loader_func = load_func_dict[data_name]
            train_data, valid_data = data_loader_func(data_path)
            # preload the dataset into memory
            train_data, valid_data = \
                [(read_im(i[0]), i[1]) for i in train_data], [(read_im(i[0]), i[1]) for i in valid_data]
            with open(cache_file_path, 'wb') as f:
                pickle.dump(
                    (train_data, valid_data,), f
                )
    else:
        data_loader_func = load_func_dict[data_name]
        train_data, valid_data = data_loader_func(data_path)

    cls_num = np.unique([i[1] for i in valid_data]).shape[0]
    if dataset_cfg.SPLIT_TRAIN:
        n_splits = dataset_cfg.N_SPLITS
        file_name = os.path.join(
            './cached_data', 'split_{:s}_N{:d}'.format(data_name.Upper(), n_splits)
        )
        if os.path.exists(file_name):
            with open(file_name, 'rb') as f:
                split_list = pickle.load(f)
        else:
            sss = sklearn.model_selection.StratifiedShuffleSplit(n_splits=n_splits)
            split_list = list(sss.split(X=[i[0] for i in train_data], y=[i[1] for i in train_data]))
            with open(file_name, 'wb') as f:
                pickle.dump(split_list, f)

        train_data, valid_data = split_list[dataset_cfg.USED_SPLIT_ID]

    if dataset_cfg.FEW_SHOT.ENABLED:
        # select shot number samples from each class to reconstruct the dataset
        train_dict, valid_dict = defaultdict(list), defaultdict(list)
        for im_path, im_label in train_data:
            train_dict[im_label].append((im_path, im_label))
        for im_path, im_label in valid_data:
            valid_dict[im_label].append((im_path, im_label))
        assert dataset_cfg.FEW_SHOT.TRAIN_SHOT > 0, "train shot should be greater than 0"
        train_data = []
        if dataset_cfg.FEW_SHOT.VALID_SHOT != -1:
            valid_data = []
        for key in train_dict.keys():
            if len(train_dict[key]) > dataset_cfg.FEW_SHOT.TRAIN_SHOT:
                train_data.extend(random.sample(train_dict[key], k=dataset_cfg.FEW_SHOT.TRAIN_SHOT))
            else:
                train_data.extend(random.choices(train_dict[key], k=dataset_cfg.FEW_SHOT.TRAIN_SHOT))
            if dataset_cfg.FEW_SHOT.VALID_SHOT != -1:
                valid_data.extend(random.sample(valid_dict[key], k=dataset_cfg.FEW_SHOT.VALID_SHOT))
    table = [
        ['SPLIT', 'SHOT', 'NUM.'],
        ['TRAIN', dataset_cfg.FEW_SHOT.TRAIN_SHOT, len(train_data)],
        ['VALID', dataset_cfg.FEW_SHOT.VALID_SHOT, len(valid_data)],
    ]
    print(tabulate(table))

    train_dataset = FineGrainedDataset(train_data, trans_func.train_trans, dataset_cfg.CACHED)
    valid_dataset = FineGrainedDataset(valid_data, trans_func.valid_trans, dataset_cfg.CACHED)
    print('Class Number :', cls_num)
    return train_dataset, valid_dataset, cls_num


def build_dataloader(fgic_dataset, dataloader_cfg, shuffle,) -> DataLoader:
    batch_size = dataloader_cfg.BATCH_SIZE
    if not shuffle:
        batch_size = int(batch_size * dataloader_cfg.VALID_SCALE)
    dataloader = DataLoader(
        fgic_dataset,
        batch_size=batch_size, shuffle=shuffle,
        pin_memory=True, num_workers=dataloader_cfg.NUM_WORKERS,
        persistent_workers=True, prefetch_factor=4,
    )
    return dataloader
