import torch
import torch.nn as nn
import torch.nn.functional as func


class DeXPaReClassifier(nn.Module):  # Cross Part Relation Classifier.
    def __init__(self, img_dim, tex_dim, feat_dim, n_prompts, cls_num, map_num, reduce=False, independent_norm=True):
        super(DeXPaReClassifier, self).__init__()
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

        self.cls_num  = cls_num
        self.n_prompt = n_prompts
        fake_num, part_num, feat_num = 16, n_prompts, tex_dim
        self.fake_cls = nn.Parameter(torch.randn(part_num, fake_num, feat_num,), requires_grad=True)
        self.fc       = nn.Linear(tex_dim, tex_dim)
        # cross-part classifier
        in_dim                      = n_prompts * map_num * fake_num
        self.prt_fc                 = nn.Sequential(
            nn.BatchNorm1d(in_dim),
            nn.Linear(in_dim, feat_dim),
            nn.ELU(),
            nn.BatchNorm1d(feat_dim),
            nn.Linear(feat_dim, cls_num),
        )
        self.x_sf                   = 64.0

        self.register_buffer(
            name='bank', tensor=torch.zeros(
                size=(fake_num, feat_num)
            ), persistent=True,
        )

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

        tmp = torch.einsum(
            'p x c, p y c -> p xy',
            self.fake_cls,  # prompt_num x fake_num x feat_dim
            self.fc(tex_f), # prompt_num x cls_num x feat_dim
        ) # prompt_num x fake_num x cls_num
        tex_f = torch.einsum(
            'p xy, p y c -> p x c', torch.softmax(tmp, dim=-1), tex_f
        ) # prompt_num x fake_num x feat_dim

        cp_rel = torch.einsum(
            'b x c, y d c -> b yxd',
            self.icomp(img_f.permute(0, 2, 1)).permute(0, 2, 1),
            self.tcomp(tex_f.permute(1, 2, 0)).permute(2, 0, 1),
        ).reshape(img_f.shape[0], self.n_prompt, -1)

        pred = self.prt_fc(cp_rel.reshape(img_f.shape[0], -1)).reshape(img_f.shape[0], cls)
        return pred
