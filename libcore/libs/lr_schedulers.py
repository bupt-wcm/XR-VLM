import copy

import math

import numpy as np
import torch
from abc import ABC, abstractmethod
from torch.optim.lr_scheduler import _LRScheduler


class BaseWarmUpScheduler(_LRScheduler, ABC):
    def __init__(self, optimizer: torch.optim.Optimizer, sche_params, warmup_params):
        self.last_epoch         = sche_params['LAST_EPOCH']
        self.verbose            = sche_params['VERBOSE']

        self.warmup_mode        = warmup_params['MODE']
        self.warmup_size        = warmup_params['SIZE']
        self.warmup_value       = warmup_params['VALUE']
        self.warmup_min_value   = warmup_params['MIN_VALUE']

        self.base_lrs = [param['lr'] for param in optimizer.param_groups]
        self._count_step = 0

        super(BaseWarmUpScheduler, self).__init__(optimizer, self.last_epoch, self.verbose)

    @abstractmethod
    def _get_scale(self, step_size):
        return NotImplementedError

    def get_warmup_lr_scale(self, step_size):
        if self.warmup_mode == 'linear':
            scale = (step_size + 1) / self.warmup_size
        else:
            scale = [self.warmup_value / v for v in self.base_lrs]
        return scale

    def step(self, step_size: int = None):
        if step_size is None:
            step_size = self._count_step
            self._count_step += 1

        scale = self._get_scale(step_size)
        if isinstance(scale, list):
            assert len(scale) == len(self.base_lrs)
            for optimizer_id, param in enumerate(self.optimizer.param_groups):
                param['lr'] = self.base_lrs[optimizer_id] * scale[optimizer_id]
        else:
            for optimizer_id, param in enumerate(self.optimizer.param_groups):
                param['lr'] = self.base_lrs[optimizer_id] * scale

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]


class WarmUpStepLR(BaseWarmUpScheduler):
    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            sche_params, warmup_params,
    ):
        self.step_size          = sche_params['STEP_SIZE']
        self.gamma              = sche_params['GAMMA']
        self.delay              = sche_params['DELAY']
        super(WarmUpStepLR, self).__init__(optimizer, sche_params, warmup_params)

    def _get_scale(self, step_size):
        if step_size < self.warmup_size:
            scale = self.get_warmup_lr_scale(step_size)
        elif step_size < self.warmup_size + self.delay:
            scale = 1.0
        else:
            scale = self.gamma ** ((step_size - self.warmup_size - self.delay + 1) // self.step_size)
        return scale


class WarmUpMilestones(BaseWarmUpScheduler):
    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            sche_params, warmup_params,
    ):

        self.milestones = sche_params['MILESTONES']
        self.gamma      = sche_params['GAMMA']
        self.delay      = sche_params['DELAY']
        sorted(self.milestones)
        super(WarmUpMilestones, self).__init__(optimizer, sche_params, warmup_params)

    def _get_scale(self, step_size):
        if step_size < self.warmup_size:
            scale = self.get_warmup_lr_scale(step_size)
        elif step_size < self.warmup_size + self.delay:
            scale = 1.0
        else:
            step_size = (step_size - self.warmup_size - self.delay + 1)
            milestones = copy.copy(self.milestones)
            milestones.append(step_size)
            milestones = sorted(milestones)
            scale = self.gamma ** milestones.index(step_size)
        return scale


class WarmUpCosineLR(BaseWarmUpScheduler):
    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            sche_params, warmup_params,
    ):
        self.t_0        = sche_params['T_0']
        self.t_multi    = sche_params['T_MULTI']
        self.delay      = sche_params['DELAY']
        super(WarmUpCosineLR, self).__init__(optimizer, sche_params, warmup_params)

    def _get_scale(self, step_size):

        if step_size < self.warmup_size:
            scale = self.get_warmup_lr_scale(step_size)
        elif step_size < self.warmup_size + self.delay:
            scale = 1.0
        else:
            step_size = step_size - self.warmup_size - self.delay
            for i in range(10 ** 8):
                if (self.t_multi ** i) * self.t_0 < step_size:
                    step_size -= (self.t_multi ** i) * self.t_0
                else:
                    break

            curr_size = (self.t_multi ** i) * self.t_0

            scale = np.cos(step_size / curr_size * math.pi) * 0.5 + 0.5
        return scale


def get_lr_scheduler(name, optimizer, sche_params, warmup_params):
    name = name.lower()
    if name == 'step':
        return WarmUpStepLR(optimizer, sche_params, warmup_params)
    elif name == 'cosine':
        return WarmUpCosineLR(optimizer, sche_params, warmup_params)
    elif name == 'miles':
        return WarmUpMilestones(optimizer, sche_params, warmup_params)
    else:
        raise KeyError('Unsupported Scheduler Type {:s}, Choosing from [step, cosine]'.format(name))
