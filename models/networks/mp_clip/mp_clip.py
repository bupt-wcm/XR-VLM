import pprint
import torch

from torch.optim import SGD
from libcore.libs.pl_model import FgICModel
from libcore.libs.registry import model_register
from libcore.libs.lr_schedulers import get_lr_scheduler

from models.utils import make_pair

from .mpnet_modified import ModifiedNet
from .single_mlp import SingleMLP
from .single_pwcs import SinglePwCS
from .single_cprm import SingleCpRM

VARIANT_DICT = {
    'default'   : ModifiedNet,
    'single_vis': SingleMLP,
    'single_sim': SinglePwCS,
    'single_xpr': SingleCpRM,
    'modified'  : ModifiedNet,
}

@model_register.register_module('mp_clip')
class MpCLIPModel(FgICModel):
    def __init__(self, model_cfg, cls_num):
        super(MpCLIPModel, self).__init__(cls_num, model_cfg)
        self.automatic_optimization = True

    def build_network(self):
        variant = self.model_cfg.NET.VARIANT
        net = VARIANT_DICT[variant](self.model_cfg.NET, self.cls_num)
        return net, net.out_num

    def backward_func(self, loss, optim):
        optim.zero_grad()
        self.manual_backward(loss)
        optim.step()

    def training_step(self, batch_data, batch_idx):
        im, label           = batch_data
        loss, losses, _     = self.network(im, label)
        self.train_step_output = {'loss': loss.item(), 'losses': torch.stack(losses, dim=0),}
        return loss

    def validation_step(self, batch_data, batch_idx):
        im, label           = batch_data
        loss, losses, preds = self.network.inference(im, label)
        return loss, torch.stack(losses, dim=0), preds

    def configure_optimizers(self):
        optim_cfg           = self.model_cfg.OPTIM
        lr                  = make_pair(optim_cfg.LR)
        wd                  = make_pair(optim_cfg.WEIGHT_DECAY)
        mt                  = make_pair(optim_cfg.MOMENTUM)

        newadd_strings      = ['classifier', 'neck', 'mp_loss', 'prompt', 'scale_factor']
        if 'RN' in self.model_cfg.NET.BACKBONE.NAME:
            # ft_strings      = ['attnpool', ]
            ft_strings      = []
        else:
            # print("module names_to_update:", list(self.network.named_modules()))
            # ft_strings      = ['ln_post', 'visual_backbone.proj', 'transformer.resblocks.11']
            ft_strings      = []
        param1_to_update    = []
        param2_to_update    = []
        names_to_update     = []
        for name, params in self.network.named_parameters():
            if any(s in name for s in ft_strings):
                param1_to_update.append(params)
                names_to_update.append([name, lr[0], params.shape])
            elif any(s in name for s in newadd_strings):
                param2_to_update.append(params)
                names_to_update.append([name, lr[1], params.shape])
            else:
                params.requires_grad = False
        total_train_params = 1
        self.print(pprint.pformat(names_to_update))
        for p in param1_to_update + param2_to_update:
            ss = 1
            for s in p.shape:
                ss *= s
            total_train_params += ss
        print('Total Trainable Params: %.2f M' % (total_train_params // 1000000))
        optimizer           = SGD(
            [
                {'params': param1_to_update, 'lr': lr[0], 'weight_decay': wd[0], 'momentum': mt[0]},
                {'params': param2_to_update, 'lr': lr[1], 'weight_decay': wd[1], 'momentum': mt[1]},
            ], lr=lr[1], weight_decay=wd[1], momentum=mt[1]
        )
        sched_cfg           = self.model_cfg.SCHED
        scheduler           = get_lr_scheduler(sched_cfg.NAME, optimizer, sched_cfg[sched_cfg.NAME], sched_cfg.WARMUP)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler, "interval": "epoch", "frequency": 1,
            }
        }