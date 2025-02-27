import torch
import torch.nn as nn
import torch.nn.functional as func

from models.utils import load_clip_to_cpu
from models.modules.tex_encoder import TexEncoder
from models.modules.prompts.mpp_prompt import MppPrompt

from models.modules.vis_encoder import ProxyVisualEncoder
from models.modules.classifier import XCoPaClassifier


class SinglePwCS(nn.Module):
    def __init__(self, net_cfg, cls_num):
        super(SinglePwCS, self).__init__()

        self.net_cfg = net_cfg
        # basic information or parameters
        clip_model              = load_clip_to_cpu(net_cfg.BACKBONE).float()
        self.dtype              = clip_model.dtype
        self.scale_factor       = clip_model.logit_scale
        self.cls_num            = cls_num

        # text branch
        self.mp_prompts         = MppPrompt(
            clip_model, net_cfg.PROMPTS.DATA_NAME.lower(), net_cfg.PROMPTS.PARAMS,
        )
        self.tokenized_prompts  = self.mp_prompts.tokenized_prompts
        self.tex_encoder        = TexEncoder(clip_model)
        tex_dim                 = self.tex_encoder.text_projection.shape[-1]
        self.n_prompts          = self.mp_prompts.n_group

        # image branch
        self.img_encoder        = ProxyVisualEncoder(net_cfg, clip_model)
        self.map_num            = net_cfg.NECK.PARAMS.N_MAPS
        img_dim                 = self.img_encoder.out_dim

        # contrastive classifier
        self.xc_classifier      = XCoPaClassifier()

        # used for accelerate the inference speed.
        self.tf_updated = False
        self.register_buffer(
            'tex_f',
            torch.zeros((net_cfg.PROMPTS.PARAMS.N_GROUP, cls_num, 512))
        )

        self.out_num = 1
        self.label_smooth       = net_cfg.LABEL_SMOOTH

    def img_feature(self, im):
        img_f = self.img_encoder(im.type(self.dtype))
        return img_f

    def tex_feature(self):
        prompts = self.mp_prompts()
        n_g, n_s, n_fle, n_fix = (
            self.mp_prompts.n_group, self.mp_prompts.n_split, self.mp_prompts.n_fle, self.mp_prompts.n_fix
        )
        prompts = prompts.reshape(n_g, self.cls_num, -1, prompts.shape[-1])
        tokenized_prompts = self.tokenized_prompts
        n_group, n_class, n_size, n_dim = prompts.shape
        flatten_prompts = prompts.reshape(-1, n_size, n_dim)
        tex_f = self.tex_encoder(flatten_prompts, tokenized_prompts).reshape(n_group, n_class, -1)
        return tex_f

    def forward(self, im, lb):
        self.tf_updated     = False

        img_f               = self.img_feature(im)
        tex_f               = self.tex_feature()
        logit_scale         = self.scale_factor.exp()

        pwcs_pred           = self.xc_classifier(img_f, tex_f, logit_scale)
        loss                = func.cross_entropy(pwcs_pred, lb, label_smoothing=self.label_smooth)

        return loss, [loss, ], [pwcs_pred, ]

    def inference(self, im, lb):
        if not self.tf_updated:
            self.tex_f      = self.tex_feature()  # side effect, set final_prompts to the tex_f
            self.tf_updated = True
        img_f               = self.img_feature(im)
        logit_scale         = self.scale_factor.exp()
        pwcs_pred           = self.xc_classifier(img_f, self.tex_f, logit_scale)
        loss                = func.cross_entropy(pwcs_pred, lb, label_smoothing=self.label_smooth)

        return loss, [loss, ], [pwcs_pred, ]
