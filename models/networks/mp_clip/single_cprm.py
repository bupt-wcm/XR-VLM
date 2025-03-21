import torch
import torch.nn as nn
import torch.nn.functional as func

from models.utils import load_clip_to_cpu
from models.modules.tex_encoder import TexEncoder
from models.modules.prompts.mpp_prompt import MppPrompt
from models.modules.prompts.hc_prompt import HcPrompt

from models.modules.vis_encoder import ProxyVisualEncoder
from models.modules.classifier import XPaReClassifier, LightXPaReClassifier

# from models.modules.classifier.cprm_plain import XPaReClassifier_b as XPaReClassifier
# from models.modules.classifier.cprm_plain import XPaReClassifier_cp as XPaReClassifier
# from models.modules.classifier.cprm_plain import XPaReClassifier_cc as XPaReClassifier


class SingleCpRM(nn.Module):
    def __init__(self, net_cfg, cls_num):
        super(SingleCpRM, self).__init__()

        self.net_cfg = net_cfg
        # basic information or parameters
        clip_model              = load_clip_to_cpu(net_cfg.BACKBONE).float()
        self.dtype              = clip_model.dtype
        self.scale_factor       = clip_model.logit_scale
        self.cls_num            = cls_num

        # text branch
        if net_cfg.PROMPTS.TYPE == 'mp_clip':
            self.mp_prompts         = MppPrompt(
                clip_model, net_cfg.PROMPTS.DATA_NAME.lower(), net_cfg.PROMPTS.PARAMS,
            )
        elif net_cfg.PROMPTS.TYPE == 'hc_clip':
            self.mp_prompts         = HcPrompt(
                clip_model, net_cfg.PROMPTS.DATA_NAME.lower(), net_cfg.PROMPTS.PARAMS,
            )
        else:
            raise NotImplementedError('Unknown Prompt Type: %s' % net_cfg.PROMPTS.TYPE)
        self.tokenized_prompts  = self.mp_prompts.tokenized_prompts
        self.tex_encoder        = TexEncoder(clip_model)
        tex_dim                 = self.tex_encoder.text_projection.shape[-1]
        self.n_prompts          = self.mp_prompts.n_group

        # image branch
        self.img_encoder        = ProxyVisualEncoder(net_cfg, clip_model)
        self.map_num            = net_cfg.NECK.PARAMS.N_MAPS
        img_dim                 = self.img_encoder.out_dim

        # shared by different classifiers and used for saving parameters or memories.
        feat_dim                = net_cfg.FC.FEAT_DIM
        reduce                  = net_cfg.FC.REDUCE
        ind_norm                = net_cfg.FC.INDEPENDENT_NORM
        de_flag                 = net_cfg.FC.DATA_EFFICIENT
        fake_num                = net_cfg.FC.FAKE_NUM
        # cross-part classifier
        self.xp_classifier      = XPaReClassifier(
            img_dim, tex_dim, feat_dim, self.n_prompts, cls_num, self.map_num,
            reduce=reduce,
            independent_norm=ind_norm,
            fake_num=fake_num,
            data_efficient=de_flag,
        )

        # used for accelerate the inference speed.
        self.tf_updated = False
        self.register_buffer('tex_f', None,)

        self.out_num = 1
        self.label_smooth       = net_cfg.LABEL_SMOOTH

    def img_feature(self, im):
        img_f = self.img_encoder(im.type(self.dtype))
        return img_f

    def tex_feature(self):
        prompts = self.mp_prompts()
        n_g = self.mp_prompts.n_group
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
        cprm_pred, con_loss = self.xp_classifier(img_f, tex_f)
        cls_loss            = func.cross_entropy(cprm_pred, lb, label_smoothing=self.label_smooth)

        loss = cls_loss + con_loss
        return loss, [cls_loss, con_loss], [cprm_pred, ]

    def inference(self, im, lb):
        if not self.tf_updated:
            self.tex_f      = self.tex_feature()  # side effect, set final_prompts to the tex_f
            self.tf_updated = True
        img_f               = self.img_feature(im)
        cprm_pred, _           = self.xp_classifier(img_f, self.tex_f)
        cls_loss            = func.cross_entropy(cprm_pred, lb, label_smoothing=self.label_smooth)
        return cls_loss, [cls_loss, ], [cprm_pred, ]
