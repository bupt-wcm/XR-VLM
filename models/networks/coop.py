import torch
import torch.nn as nn
import torch.nn.functional as func

from libcore.libs.pl_model import FgICModel
from libcore.libs.registry import model_register
from torch.optim import SGD
from libcore.libs.lr_schedulers import get_lr_scheduler

from models.utils import load_clip_to_cpu
from models.modules.tex_encoder import TexEncoder
from models.modules.prompts import CoOpPrompt


class CoOpCLIP(nn.Module):
    def __init__(self, net_cfg):
        super(CoOpCLIP, self).__init__()

        clip_model = load_clip_to_cpu(net_cfg.BACKBONE)

        self.dtype = clip_model.dtype
        # produce triplet learnable prompts for text encoder
        self.coop_prompts = CoOpPrompt(
            clip_model,
            net_cfg.PROMPTS.DATA_NAME.lower(),
            net_cfg.PROMPTS.PARAMS,
        )
        self.tokenized_prompts = self.coop_prompts.tokenized_prompts
        self.tex_encoder = TexEncoder(clip_model)

        self.img_encoder = clip_model.visual
        self.logit_scale = clip_model.logit_scale

    def forward(self, im):
        img_f = self.img_encoder(im.type(self.dtype))
        prompts = self.coop_prompts()
        tokenized_prompts = self.tokenized_prompts
        tex_f = self.tex_encoder(prompts, tokenized_prompts)
        print(img_f.shape, tex_f.shape)
        img_f = img_f / img_f.norm(dim=-1, keepdim=True)
        tex_f = tex_f / tex_f.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * img_f @ tex_f.t()

        return logits


@model_register.register_module('coop_clip')
class CoOpCLIPModel(FgICModel):
    def __init__(self, model_cfg, cls_num):
        super(CoOpCLIPModel, self).__init__(cls_num, model_cfg)
        self.automatic_optimization = False

    def build_network(self):
        net = CoOpCLIP(self.model_cfg.NET)
        for name, params in net.named_parameters():
            if 'coop_prompts' not in name:
                params.requires_grad_(False)
        return net, 1

    def backward_func(self, loss, optim):
        optim.zero_grad()
        self.manual_backward(loss)
        if self.model_cfg.OPTIM.clip_enabled:
            torch.nn.utils.clip_grad_norm_(
                self.network.parameters(),
                max_norm=self.model_cfg.OPTIM.clip_val, norm_type=self.model_cfg.OPTIM.norm_type,
            )
        optim.step()

    def training_step(self, batch_data, batch_idx):
        optim = self.optimizers(use_pl_optimizer=True)
        if isinstance(optim, list):
            optim = optim[0]
        self.network.img_encoder.eval()
        im, label = batch_data
        y = self.network(im)
        cls_loss = func.cross_entropy(y, label, label_smoothing=0.0)
        # optimization
        self.train_step_output = {
            'loss': cls_loss.item(), 'losses': torch.stack([cls_loss, ], dim=0)
        }
        self.backward_func(cls_loss, optim)

        sch = self.lr_schedulers()
        sch.step(self.trainer.current_epoch)
        # self.print(self.current_epoch, sch.get_last_lr())

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
                {'params': self.network.coop_prompts.parameters()},
            ], lr=lr, weight_decay=wd, momentum=mt,
        )
        sched_cfg = self.model_cfg.SCHED
        scheduler = get_lr_scheduler(sched_cfg.NAME, optimizer, sched_cfg[sched_cfg.NAME], sched_cfg.WARMUP)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
