import torch
import torch.nn as nn


class XPaReClassifier(nn.Module):  # Cross Part Relation Classifier.
    def __init__(self, img_dim, tex_dim, feat_dim, n_prompts, cls_num, map_num, reduce=False, independent_norm=True):
        super(XPaReClassifier, self).__init__()
        self.independent_norm       = independent_norm
        if reduce:
            fd = img_dim // 2
            self.icomp              = nn.Sequential(
                nn.BatchNorm1d(img_dim), nn.Conv1d(img_dim, fd, 1), nn.ELU(),
            )
            self.tcomp              = nn.Sequential(
                nn.BatchNorm1d(tex_dim), nn.Conv1d(tex_dim, fd, 1), nn.ELU(),
            )
        else:
            assert tex_dim          == img_dim, (
                    'When no use reduce, the image (%d) and text (%d) dim should be same.' % (img_dim, tex_dim)
            )
            self.icomp              = nn.Identity()
            self.tcomp              = nn.Identity()

        # cross-part classifier
        in_dim                      = n_prompts * map_num * cls_num
        self.prt_fc                 = nn.Sequential(
            nn.BatchNorm1d(in_dim),
            nn.Linear(in_dim, feat_dim),
            nn.ELU(),
            nn.BatchNorm1d(feat_dim),
            nn.Linear(feat_dim, cls_num),
        )

        self.x_sf                   = 64.0

    def forward(self, img_f, tex_f):
        n, cls, c = tex_f.shape
        if self.independent_norm:
            img_f = img_f / (img_f.norm(dim=-1, keepdim=True) + 1e-6) * self.x_sf  # batch_size x map_num x c
            tex_f = tex_f / (tex_f.norm(dim=-1, keepdim=True) + 1e-6) * self.x_sf  # map_num x cls_num x c
        else:
            b, n, c = img_f.shape
            img_f = img_f.reshape(b, -1) / (img_f.reshape(b, -1).norm(dim=-1, keepdim=True) + 1e-6)
            img_f = img_f.reshape(b, n, c) * self.x_sf
            tex_f = tex_f.permute(1, 0, 2)
            tex_f = tex_f.reshape(cls, -1) / (tex_f.reshape(cls, -1).norm(dim=-1, keepdim=True) + 1e-6)
            tex_f = tex_f.reshape(cls, n, c).permute(1, 0, 2) * self.x_sf

        cp_rel = torch.einsum(
            'b x c, y d c -> b yxd',
            self.icomp(img_f.permute(0, 2, 1)).permute(0, 2, 1),
            self.tcomp(tex_f.permute(1, 2, 0)).permute(2, 0, 1),
        ).reshape(img_f.shape[0], cls, -1)

        pred = self.prt_fc(cp_rel.reshape(img_f.shape[0], -1))
        return pred
