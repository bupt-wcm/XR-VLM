import os
from collections import defaultdict
import copy
import yaml

from typing import Any
from yacs.config import CfgNode as _CfgNode

BASE_KEY = "_BASE_"


class CfgNode(_CfgNode):
    def __init__(self, init_dict=None, key_list=None):
        super().__init__(init_dict, key_list, True)

    @staticmethod
    def load_yaml_with_base(filename: str):

        with open(filename, "r") as f:
            cfg = yaml.safe_load(f)

        def merge_a_into_b(a, b):
            # merge dict a into dict b. values in a will overwrite b.
            for k, v in a.items():
                if isinstance(v, dict) and k in b:
                    assert isinstance(
                        b[k], dict
                    ), "Cannot inherit key '{}' from base!".format(k)
                    merge_a_into_b(v, b[k])
                else:
                    b[k] = v

        if BASE_KEY in cfg:
            base_cfg_file = cfg[BASE_KEY]
            if base_cfg_file.startswith("~"):
                base_cfg_file = os.path.expanduser(base_cfg_file)
            if not any(
                    map(base_cfg_file.startswith, ["/", "https://", "http://"])
            ):
                # the path to base cfg is relative to the config file itself.
                base_cfg_file = os.path.join(os.path.dirname(filename), base_cfg_file)
            base_cfg = CfgNode.load_yaml_with_base(base_cfg_file, )
            del cfg[BASE_KEY]

            merge_a_into_b(cfg, base_cfg)
            return base_cfg
        return cfg

    def merge_from_file(self, cfg_filename: str):
        """
        Merge configs from a given yaml file.
        Args:
            cfg_filename: the file name of the yaml config.
        """
        loaded_cfg = CfgNode.load_yaml_with_base(cfg_filename, )
        loaded_cfg = type(self)(loaded_cfg)
        # print(loaded_cfg)
        self.merge_from_other_cfg(loaded_cfg)

    # Forward the following calls to base, but with a check on the BASE_KEY.
    def merge_from_other_cfg(self, cfg_other):
        """
        Args:
            cfg_other (CfgNode): configs to merge from.
        """
        # print('merge from cfg:', cfg_other)
        assert (
                BASE_KEY not in cfg_other
        ), "The reserved key '{}' can only be used in files!".format(BASE_KEY)
        return super().merge_from_other_cfg(cfg_other)

    def merge_from_list(self, cfg_list: list):
        """
        Args:
            cfg_list (list): list of configs to merge from.
        """
        # print('merge from list:', cfg_list)
        keys = set(cfg_list[0::2])
        assert (
                BASE_KEY not in keys
        ), "The reserved key '{}' can only be used in files!".format(BASE_KEY)
        return super().merge_from_list(cfg_list)

    def __setattr__(self, name: str, val: Any):
        if name.startswith("COMPUTED_"):
            if name in self:
                old_val = self[name]
                if old_val == val:
                    return
                raise KeyError(
                    "Computed attributed '{}' already exists "
                    "with a different value! old={}, new={}.".format(
                        name, old_val, val
                    )
                )
            self[name] = val
        else:
            super().__setattr__(name, val)


def get_cfg(name=None):
    if name is None:
        from .default import _C, _U
        return _C.clone(), _U
    else:  # used for data config
        from .default import _DATA_CFG_DICT
        return _DATA_CFG_DICT[name.upper()].clone(), None


class Updater:
    def __init__(self, ):
        self._map = defaultdict(list)

    def update(self, cfg):
        for from_name in self._map:
            for to_names in self._map[from_name]:
                for to_name in to_names:
                    _d = cfg.clone()
                    for key in from_name.split(".")[:-1]:
                        if key not in _d:
                            raise KeyError("{} is not a valid config path".format(from_name))
                        _d = _d[key]
                    from_node = _d
                    from_key = from_name.split(".")[-1]

                    _d = cfg
                    for key in to_name.split(".")[:-1]:
                        if key not in _d:
                            raise KeyError("{} is not a valid config path".format(from_name))
                        _d = _d[key]
                    to_node = _d
                    to_key = to_name.split(".")[-1]

                    to_node[to_key] = from_node[from_key]
                    print('Update %s from %s to %s' % (to_name, from_name, from_node[from_key]))
        return cfg

    def set_dependent(self, to_name, from_name):
        self._map[from_name].append([to_name,])
