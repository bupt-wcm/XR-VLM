import torch
import torch.nn as nn

from models.utils import load_clip_to_cpu
from models.modules.tex_encoder import TexEncoder
from models.modules.prompts.mpp_prompt import MppPrompt

from models.modules.vis_encoder import ProxyVisualEncoder
from models.modules.classifier import XPaReClassifier, ImPaClassifier, XCoPaClassifier
from models.modules.loss_func import MpConLossModule


class ModifiedNet(nn.Module):
    def __init__(self, net_cfg, cls_num):
        super(ModifiedNet, self).__init__()

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

        # shared by different classifiers and used for saving parameters or memories.
        feat_dim                = net_cfg.FC.FEAT_DIM
        # cross-part classifier
        self.xp_classifier      = XPaReClassifier(
            img_dim, tex_dim, feat_dim, self.n_prompts, cls_num, self.map_num, reduce=True)
        # direct image classifier
        self.ip_classifier      = ImPaClassifier(img_dim, feat_dim, cls_num, self.map_num)
        # contrastive classifier
        self.xc_classifier      = XCoPaClassifier()
        self.mp_loss            = MpConLossModule(img_dim, tex_dim)

        # used for accelerate the inference speed.
        self.tf_updated = False
        self.register_buffer(
            'tex_f',
            torch.zeros((net_cfg.PROMPTS.PARAMS.N_GROUP, cls_num, 512))
        )

        self.out_num = 3

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

        fake_tf1, fake_tf2 = None, None
        fake_tf = tex_f.detach()
        # generate random text features
        mean, std = torch.mean(fake_tf, dim=(0, 1), keepdim=True), torch.std(fake_tf, dim=(0, 1), keepdim=True)
        fake_tf1 = torch.randn_like(fake_tf) * std + mean
        # swap different features from different classes.
        r_idx = torch.randperm(n_group * n_class)
        fake_tf2 = fake_tf.reshape(n_group *  n_class, -1)[r_idx, :].reshape(n_group, n_class, -1).contiguous()

        return tex_f, fake_tf1, fake_tf2

    def forward(self, im, lb):
        self.tf_updated     = False

        img_f               = self.img_feature(im)
        tex_f, ftf1, ftf2   = self.tex_feature()
        logit_scale         = self.scale_factor.exp()

        mlps_pred           = self.ip_classifier(img_f)
        pwcs_pred           = self.xc_classifier(img_f, tex_f, logit_scale)
        cprm_pred           = self.xp_classifier(img_f, tex_f)

        # counterfactual learning
        fcp1, fcp2 = None, None
        if ftf1 is not None:
            self.xp_classifier.train(mode=False)  # fix the bn in classifier
            fcp1                = self.xp_classifier(img_f.detach(), ftf1)
            fcp2                = self.xp_classifier(img_f.detach(), ftf2)
            self.xp_classifier.train(mode=True)
        # multi-part constraint learning
        con_loss            = self.mp_loss(img_f, tex_f, lb)

        return [mlps_pred, pwcs_pred, cprm_pred], [fcp1, fcp2, con_loss]

    def inference(self, im):
        if not self.tf_updated:
            self.tex_f, _, _  = self.tex_feature()  # side effect, set final_prompts to the tex_f
            self.tf_updated = True
        img_f               = self.img_feature(im)
        logit_scale         = self.scale_factor.exp()
        mlps_pred           = self.ip_classifier(img_f)
        pwcs_pred           = self.xc_classifier(img_f, self.tex_f, logit_scale)
        cprm_pred           = self.xp_classifier(img_f, self.tex_f)
        return mlps_pred, pwcs_pred, cprm_pred
