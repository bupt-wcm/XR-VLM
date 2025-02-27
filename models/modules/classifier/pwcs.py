import torch
import torch.nn as nn


class XCoPaClassifier(nn.Module):
    def __init__(self, independent_norm=True):
        super(XCoPaClassifier, self).__init__()
        self.independent_norm = independent_norm

    def forward(self, img_f, tex_f, logit_scale):
        if self.independent_norm:
            img_f = img_f / (img_f.norm(dim=-1, keepdim=True) + 1e-6)  # batch_size x map_num x c
            tex_f = tex_f / (tex_f.norm(dim=-1, keepdim=True) + 1e-6)  # map_num x cls_num x c
            cnt = max(img_f.shape[1], tex_f.shape[0])
            cos_pred = torch.einsum(
                'b n c, n d c -> b d', img_f, tex_f
            ) * logit_scale / cnt
        else:
            b, n, c = img_f.shape
            img_f = img_f.reshape(b, -1) / (img_f.reshape(b, -1).norm(dim=-1, keepdim=True) + 1e-6)
            img_f = img_f.reshape(b, n, c)

            n, cls, c = tex_f.shape
            tex_f = tex_f.permute(1, 0, 2)
            tex_f = tex_f.reshape(cls, -1) / (tex_f.reshape(cls, -1).norm(dim=-1, keepdim=True) + 1e-6)
            tex_f = tex_f.reshape(cls, n, c).permute(1, 0, 2)
            cos_pred = torch.einsum(
                'b n c, n d c -> b d', img_f, tex_f
            ) * logit_scale
        return cos_pred
