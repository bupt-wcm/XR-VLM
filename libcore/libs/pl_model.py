import abc
import functools
import time

import pytorch_lightning as pl
import torch.nn as nn
from torch.optim import SGD
from torchmetrics import Accuracy

from libcore.libs.lr_schedulers import get_lr_scheduler


class FgICModel(pl.LightningModule):
    def __init__(self, cls_num, model_cfg):
        super(FgICModel, self).__init__()

        self.cls_num = cls_num
        self.model_cfg = model_cfg
        self.network, self.out_num = self.build_network()
        self._load_from_resume()

        self.best_acc = 0.0
        self.print_freq = 10  # change this if the batch size is too large or small
        self.automatic_optimization = True  # disable manual backward
        self.valid_accuracy = nn.ModuleList(
            [
                Accuracy(task='multiclass', num_classes=cls_num) for _ in range(self.out_num)
            ]
        )

        self.train_step_output = None
        self.valid_output = []

    @abc.abstractmethod
    def build_network(self):
        pass

    def _load_from_resume(self, resume=None):
        if resume is not None:
            self.load_from_checkpoint(checkpoint_path=resume, map_location='cpu')
        elif self.model_cfg.RESUME is not None:
            self.load_from_checkpoint(self.model_cfg.RESUME, map_location='cpu')

    def configure_optimizers(self):
        optim_cfg = self.model_cfg.OPTIM
        backbone_params = dict(filter(lambda x: 'backbone' in x[0], self.network.named_parameters()))
        addition_params = dict(filter(lambda x: 'backbone' not in x[0], self.network.named_parameters()))

        self.print(
            'abnormal parameters in training',
            set(dict(self.network.named_parameters()).keys()).difference(
                set(backbone_params.keys()).union(set(addition_params.keys()))
            )
        )
        self.print(
            'addition parameters in training', addition_params.keys()
        )
        if optim_cfg.NO_BIAS_DECAY:
            backbone_bias = dict(filter(lambda x: 'bias' not in x[0] and 'bn' not in x[0], backbone_params.items()))
            backbone_params = dict(filter(lambda x: 'bias' in x[0] or 'bn' in x[0], backbone_params.items()))
            addition_bias = dict(filter(lambda x: 'bias' not in x[0] and 'bn' not in x[0], addition_params.items()))
            addition_params = dict(filter(lambda x: 'bias' in x[0] or 'bn' in x[0], addition_params.items()))

            optimizer = SGD(
                [
                    {'params': backbone_bias.values(), 'lr': optim_cfg.LR[0], 'weight_decay': 0.0},
                    {'params': backbone_params.values(), 'lr': optim_cfg.LR[1],
                     'weight_decay': optim_cfg.WEIGHT_DECAY[1]},
                    {'params': addition_bias.values(), 'lr': optim_cfg.LR[0], 'weight_decay': 0.0},
                    {'params': addition_params.values(), 'lr': optim_cfg.LR[1],
                     'weight_decay': optim_cfg.WEIGHT_DECAY[1]},
                ], lr=0.1, momentum=0.9, weight_decay=1e-4,
            )
        else:
            backbone_params = backbone_params.values()
            addition_params = addition_params.values()

            optimizer = SGD(
                [
                    {'params': backbone_params, 'lr': optim_cfg.LR[0], 'weight_decay': optim_cfg.WEIGHT_DECAY[0]},
                    {'params': addition_params, 'lr': optim_cfg.LR[1], 'weight_decay': optim_cfg.WEIGHT_DECAY[1]},
                ], lr=0.1, momentum=0.9, weight_decay=1e-4,
            )
        sched_cfg = optim_cfg.SCHED
        scheduler = get_lr_scheduler(sched_cfg.NAME, optimizer, sched_cfg[sched_cfg.NAME].PARAMS)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def on_train_batch_end(self, step_output, batch, batch_idx: int) -> None:
        step_output = self.train_step_output
        loss, losses = step_output['loss'], step_output['losses']
        if self.global_step % self.print_freq == 0:
            self.log_dict(dict([('L-%d' % idx, item) for idx, item in enumerate(losses)]), prog_bar=True)
            if self.local_rank == 0:
                print_info = time.strftime("%b%d-%H:%M:%S", time.localtime(time.time()))
                print_info += ' Epoch {:3d}|{:3d} | '.format(self.current_epoch, self.trainer.max_epochs)
                print_info += ' Iters {:03d}|{:3d} | '.format(batch_idx, self.trainer.num_training_batches)
                print_info += 'Train Loss {:8.3f} | {:s} '.format(
                    loss, statics_to_string(losses.detach().cpu().numpy().tolist(), num=1, prefix='L'),
                )
                self.print(print_info)

    def on_validation_batch_end(
            self, step_out, batch, batch_idx: int, dataloader_idx: int=0,
    ) -> None:
        loss, losses, predictions = step_out
        for idx, pred in enumerate(predictions):
            self.valid_accuracy[idx].update(pred, batch[1])
        self.valid_output.append(step_out)

    def on_validation_epoch_end(self, ) -> None:
        # record accuracy
        outputs = self.valid_output
        acc_dict = dict([("A-%d" % idx, acc.compute()) for idx, acc in enumerate(self.valid_accuracy)])
        self.log_dict(acc_dict, sync_dist=True)
        if acc_dict["A-%d" % (len(self.valid_accuracy) - 1)] > self.best_acc:
            self.best_acc = acc_dict["A-%d" % (len(self.valid_accuracy) - 1)]
        self.log('BEST_ACC', self.best_acc, sync_dist=True)

        for v in self.valid_accuracy:
            v.reset()

        # record validation losses
        num_val_batches = self.trainer.num_val_batches[-1]

        valid_loss = functools.reduce(lambda x, y: x + y, [i for i, _, _ in outputs])
        multi_loss = functools.reduce(lambda x, y: x + y, [i for _, i, _ in outputs])

        print_info = time.strftime("%b%d-%H:%M:%S", time.localtime(time.time()))
        print_info += ' Epoch {:3d}|{:3d} | '.format(self.current_epoch, self.trainer.max_epochs)
        # print_info += self.train_info + ' | '

        print_info += 'Valid Loss {:8.3f} | {:s} | {:s}'.format(
            valid_loss / num_val_batches,
            statics_to_string(multi_loss.detach().cpu().numpy().tolist(), num_val_batches, prefix='L'),
            statics_to_string(list(acc_dict.values()), 1, prefix='A', scale=100.)
        )

        print_info += ' |Best-A \033[0;32;40m{:.3f}\033[0m'.format(self.best_acc * 100.)
        self.print(print_info)

        self.valid_output.clear()


def statics_to_string(statics, num, prefix, scale=1.0):
    if isinstance(prefix, str):
        f = prefix + '{:01d} {:5.3f}'
        ss = []
        for idx, l in enumerate(statics):
            ss.append(f.format(idx, l / num * scale))
    else:
        assert isinstance(prefix, list) and len(prefix) == len(statics), \
            'The num of prefix should equal to the number of statics if the prefix is a list'
        ss = []
        for idx, (p, s) in enumerate(zip(prefix, statics)):
            f = p + '{:s}{:01d} {:5.3f}'
            ss.append(f.format(idx, s / num * scale))
    return '|'.join(ss)
