import torch
import torch.nn as nn
import torch.nn.functional as func

from libcore.libs.pl_model import FgICModel
from libcore.libs.registry import model_register
from libcore.models.backbones import build_backbone


class BaseNetArch(nn.Module):
    def __init__(self, net_cfg, cls_num):
        super(BaseNetArch, self).__init__()

        self.backbone = build_backbone(
            name=net_cfg.BACKBONE.NAME, model_type=net_cfg.BACKBONE.TYPE,
        )
        out_dim = self.backbone.out_dims[-1]

        self.classifier = nn.Sequential(
            nn.AdaptiveMaxPool2d(output_size=(1, 1)),
            nn.Flatten(-3, -1),
            nn.BatchNorm1d(out_dim),
            nn.Linear(out_dim, out_dim // 2),
            nn.BatchNorm1d(out_dim // 2),
            nn.ELU(),
            nn.Linear(out_dim // 2, cls_num)
        )

    def forward(self, im):
        x, = self.backbone(im, out_ids=(3,))
        p = self.classifier(x)
        return p


@model_register.register_module('base')
class BaseModel(FgICModel):
    def __init__(self, model_cfg, cls_num):
        super(BaseModel, self).__init__(cls_num, model_cfg)
        self.automatic_optimization = False
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    def build_network(self):
        return BaseNetArch(self.model_cfg.NET, self.cls_num)

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

        loss = self.criterion(y, label)
        self.train_step_output = {
            'loss': loss.item(), 'losses': torch.stack([
                func.cross_entropy(y, label),
            ], dim=0)
        }
        self.backward_func(loss, optim)

        sch = self.lr_schedulers()
        sch.step(self.trainer.current_epoch)

    def validation_step(self, batch_data, batch_idx):
        im, label = batch_data
        y = self.network(im)
        loss = func.cross_entropy(y, label)
        return loss, torch.stack([loss, ], dim=0), [y, ]  # gradient loss, all loss, predictions
