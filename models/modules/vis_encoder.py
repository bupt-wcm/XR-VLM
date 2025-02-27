import torch
import torch.nn as nn
import torch.nn.functional as func

from models.modules.uni_atte import UniAttPooling
from torch.utils.checkpoint import checkpoint_sequential


def build_neck(net_cfg, img_dim):
    if net_cfg.NECK.TYPE == 'NAI':
        neck = None
    elif net_cfg.NECK.TYPE == 'AVG':
        neck = nn.AdaptiveAvgPool1d(output_size=(1,))
    elif net_cfg.NECK.TYPE == 'MAX':
        neck = nn.AdaptiveMaxPool1d(output_size=(1,))
    elif net_cfg.NECK.TYPE == 'UNI_ATT':
        neck = UniAttPooling(
            in_dim=img_dim,
            map_num=net_cfg.NECK.PARAMS.N_MAPS,
            softmax_scale=net_cfg.NECK.PARAMS.SM_SCALE,
        )
    else:
        raise NotImplementedError('Unknown NECK type: %s' % net_cfg.NECK.TYPE)
    return neck


class ProxyVisualEncoder(nn.Module):
    def __init__(self, net_cfg, clip_model):
        super(ProxyVisualEncoder, self).__init__()
        self.visual_backbone    = clip_model.visual
        self.backbone_type      = net_cfg.BACKBONE.NAME
        self.visual_prompt      = net_cfg.BACKBONE.VISUAL_PROMPT
        back_dim                = 2048 if 'RN' in net_cfg.BACKBONE.NAME else 768
        self.neck               = build_neck(net_cfg, back_dim)  # keep simple, make dim same.
        self.out_dim            = clip_model.text_projection.shape[-1]
        self.pool_type          = net_cfg.NECK.TYPE

        """
        if 'RN' in net_cfg.BACKBONE.NAME:
            for m in self.visual_backbone.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.track_running_stats = False
        """
        self.prompt_emb         = None
        if self.visual_prompt:
            prompt_cfg = net_cfg.PROMPTS.PARAMS
            ctx_vision_vectors = torch.empty(prompt_cfg.N_GROUP, prompt_cfg.N_FLE, 768, dtype=clip_model.dtype)
            nn.init.normal_(ctx_vision_vectors, std=0.02)
            self.prompt_emb = nn.Parameter(ctx_vision_vectors)  # parameters of vision prompt to be learned

    def res_forward(self, x):
        bbn = self.visual_backbone
        def stem(xz):
            xz = bbn.relu1(bbn.bn1(bbn.conv1(xz)))
            xz = bbn.relu2(bbn.bn2(bbn.conv2(xz)))
            xz = bbn.relu3(bbn.bn3(bbn.conv3(xz)))
            xz = bbn.avgpool(xz)
            return xz
        x = x.type(bbn.conv1.weight.dtype)
        x = stem(x)
        x = bbn.layer1(x)
        x = bbn.layer2(x)
        x = bbn.layer3(x)
        x = bbn.layer4(x)

        b, _, h, w = x.shape
        x1 = x

        # processing without self-attention
        embed_dim = x.shape[1]
        pos_emb = bbn.attnpool.positional_embedding
        mean_pos_emb, patch_pos_emb = pos_emb[0:1], pos_emb[1:]
        pos_emb = torch.cat(
            [
                mean_pos_emb,
                func.interpolate(
                    patch_pos_emb.reshape(
                        1, bbn.attnpool.spacial_dim, bbn.attnpool.spacial_dim, embed_dim
                    ).permute(0, 3, 1, 2),
                    size = (x.shape[2], x.shape[3]),
                    mode = 'bilinear',
                ).permute(2, 3, 0, 1).reshape(-1, embed_dim),
            ], dim=0,
        )
        
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + pos_emb[:, None, :].to(x.dtype)  # (HW+1)NC
        f = bbn.attnpool.c_proj(bbn.attnpool.v_proj(x))

        # processing with self-attention
        # f = bbn.attnpool(x)

        f = f.permute(1, 2, 0)[:, :, 1:].reshape(b, -1, h, w).contiguous()
        return x1, f

    def vit_forward(self, x):
        bbn = self.visual_backbone
        b = x.shape[0]
        x = bbn.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [bbn.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]

        x = x + bbn.positional_embedding.to(x.dtype)

        prompt = self.prompt_emb
        if self.visual_prompt:
            batch_size, num_vis_prompt = x.shape[0], prompt.shape[0]
            vis_prompt_len = prompt.shape[1]
            grid_size, width = x.shape[1:]
            x = x.unsqueeze(0).expand(num_vis_prompt, -1, -1, -1)
            visual_ctx = prompt.unsqueeze(1).expand(-1, batch_size, -1, -1)
            x = torch.cat([x, visual_ctx], dim=2)
            x = x.contiguous().view(num_vis_prompt * batch_size, grid_size + vis_prompt_len, width)

        x = bbn.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = checkpoint_sequential(bbn.transformer.resblocks[:-2], segments=1, input=x, use_reentrant=False)
        f = bbn.transformer.resblocks[-2:](x)
        f = f.permute(1, 0, 2)  # LND -> NLD
        f = bbn.ln_post(f)
        if self.visual_prompt:
            f = f.contiguous().view(prompt.shape[0], b, grid_size + vis_prompt_len, width)
            f = f.permute(1, 0, 2, 3).reshape(b, -1, width)
            x = x.permute(1, 0, 2).reshape(
                prompt.shape[0], b, grid_size + vis_prompt_len, width
            ).permute(1, 0, 2, 3).reshape(b, -1, width).permute(1, 0, 2).contiguous()

        if bbn.proj is not None:
            f = f @ bbn.proj
        return x.permute(1, 0, 2), f

    def forward(self, im):
        if 'RN' in self.backbone_type:
            x, f = self.res_forward(im)  # x shape: B x C x H x W, f shape: B x C x H x W
            b, c, h, w = x.shape
            f_c = f.shape[1]
            if 'AVG' in self.pool_type:
                return func.adaptive_avg_pool2d(f, 1).reshape(b, 1, f_c)
            elif 'MAX' in self.pool_type:
                return func.adaptive_max_pool2d(f, 1).reshape(b, 1, f_c)
            elif 'ATT' in self.pool_type:
                x = x.reshape(*x.shape[:2], -1).contiguous()
                return self.neck(x, f.flatten(start_dim=2))  # input B x C x (HW), output: B x N x C
            else:
                return None
        elif 'vit' in self.backbone_type.lower():
            x, f = self.vit_forward(im)
            if 'AVG' in self.pool_type:
                return func.adaptive_avg_pool1d(f[:, 1:].permute(0, 2, 1), output_size=1).squeeze().unsqueeze(1)
            elif 'MAX' in self.pool_type:
                return func.adaptive_max_pool2d(f[:, 1:].permute(0, 2, 1), output_size=1).squeeze().unsqueeze(1)
            elif 'ATT' in self.pool_type:
                return self.neck(x.permute(0, 2, 1), f.permute(0, 2, 1))  # input B x C x N, output: B x N x C
            else:
                return f[:, :1,]
        else:
            raise KeyError('Model type not supported.')
