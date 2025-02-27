import torch
import torch.nn as nn
import torch.nn.functional as func

from libcore.libs.pl_model import FgICModel
from libcore.libs.registry import model_register
from torch.optim import SGD
from libcore.libs.lr_schedulers import get_lr_scheduler

from models.utils import load_clip_to_cpu
from models.modules.tex_encoder import TexEncoder
from models.modules.prompts import BasePrompts



class ZeroShotCLIP(nn.Module):
    def __init__(self, net_cfg, cls_num):
        super(ZeroShotCLIP, self).__init__()

        clip_model = load_clip_to_cpu(net_cfg.BACKBONE).float()

        self.dtype = clip_model.dtype
        # produce triplet learnable prompts for text encoder
        self.prompts = BasePrompts(clip_model, net_cfg.PROMPTS.DATA_NAME.lower())
        self.tokenized_prompts = self.prompts.tokenized_prompts
        self.tex_encoder = TexEncoder(clip_model)

        self.img_encoder = clip_model.visual
        self.logit_scale = clip_model.logit_scale

    def forward(self, im):
        img_f, img_x = self.img_encoder(im.type(self.dtype))[:, 0]

        prompts = self.prompts()
        tokenized_prompts = self.tokenized_prompts
        tex_f = self.tex_encoder(prompts, tokenized_prompts)
        img_f = img_f / img_f.norm(dim=-1, keepdim=True)
        tex_f = tex_f / tex_f.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * img_f @ tex_f.t()

        return logits


@model_register.register_module('zs_clip')
class ZeroShotCLIPModel(FgICModel):
    def __init__(self, model_cfg, cls_num):
        super(ZeroShotCLIPModel, self).__init__(cls_num, model_cfg)
        self.automatic_optimization = False

    def build_network(self):
        return ZeroShotCLIP(self.model_cfg.NET, self.cls_num), 1

    def backward_func(self, loss, optim):
        optim.zero_grad()
        self.manual_backward(loss)
        optim.step()

    def training_step(self, batch_data, batch_idx):
        optim = self.optimizers(use_pl_optimizer=True)
        if isinstance(optim, list):
            optim = optim[0]

        im, label = batch_data

        y = self.network(im)
        cls_loss = func.cross_entropy(y, label, label_smoothing=0.1)

        # optimization
        self.train_step_output = {
            'loss': cls_loss.item(), 'losses': torch.stack([cls_loss, ], dim=0)
        }
        self.backward_func(cls_loss, optim)

        sch = self.lr_schedulers()
        sch.step(self.trainer.current_epoch)

    def validation_step(self, batch_data, batch_idx):
        im, label = batch_data
        y = self.network(im)
        # print(y.shape, label.shape)
        loss = func.cross_entropy(y, label)
        return loss, torch.stack([loss, ], dim=0), [y, ]  # gradient loss, all loss, predictions

    def configure_optimizers(self):
        optim_cfg = self.model_cfg.OPTIM
        lr = optim_cfg.LR[0] if isinstance(optim_cfg.LR, list) else optim_cfg.LR
        wd = optim_cfg.WEIGHT_DECAY[0] if isinstance(optim_cfg.WEIGHT_DECAY, list) else optim_cfg.WEIGHT_DECAY
        mt = optim_cfg.MOMENTUM[0] if isinstance(optim_cfg.MOMENTUM, list) else optim_cfg.MOMENTUM

        optimizer = SGD(
            [
                {'params': self.network.img_encoder.parameters()},
            ], lr=lr, momentum=wd, weight_decay=mt,
        )
        sched_cfg = self.model_cfg.SCHED
        scheduler = get_lr_scheduler(sched_cfg.NAME, optimizer, sched_cfg[sched_cfg.NAME].PARAMS)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
