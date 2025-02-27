import torch
import torch.nn as nn
import torch.nn.functional as func

from models.utils import load_clip_to_cpu
from models.modules.vis_encoder import ProxyVisualEncoder
from models.modules.classifier.mlp import ImPaClassifier


class SingleMLP(nn.Module):
    def __init__(self, net_cfg, cls_num):
        super(SingleMLP, self).__init__()

        self.net_cfg = net_cfg
        # basic information or parameters
        clip_model              = load_clip_to_cpu(net_cfg.BACKBONE).float()
        self.dtype              = clip_model.dtype
        self.cls_num            = cls_num

        # image branch
        self.img_encoder        = ProxyVisualEncoder(net_cfg, clip_model)
        self.map_num            = net_cfg.NECK.PARAMS.N_MAPS
        img_dim                 = self.img_encoder.out_dim

        feat_dim                = net_cfg.FC.FEAT_DIM
        self.ip_classifier      = ImPaClassifier(img_dim, feat_dim, cls_num, self.map_num)

        self.out_num            = 1
        self.label_smooth       = net_cfg.LABEL_SMOOTH

    def img_feature(self, im):
        img_f = self.img_encoder(im.type(self.dtype))
        return img_f

    def forward(self, im, lb):
        img_f               = self.img_feature(im)
        mlps_pred           = self.ip_classifier(img_f)
        loss                = func.cross_entropy(mlps_pred, lb, label_smoothing=self.label_smooth)

        return loss, [loss, ], [mlps_pred, ]

    def inference(self, im, lb):
        img_f               = self.img_feature(im)
        mlps_pred           = self.ip_classifier(img_f)
        loss                = func.cross_entropy(mlps_pred, lb, label_smoothing=self.label_smooth)

        return loss, [loss, ], [mlps_pred, ]
