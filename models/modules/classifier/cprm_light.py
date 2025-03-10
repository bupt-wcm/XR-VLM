import torch
import torch.nn as nn
import torch.nn.functional as func


EPSILON = 1e-6


class XPaReClassifier(nn.Module):  # Cross Part Relation Classifier.
    def __init__(
            self,
            img_dim, tex_dim, feat_dim, n_prompts, cls_num, map_num,
            reduce=False, independent_norm=True,
            fake_num=16, data_efficient=True,
    ):
        super(XPaReClassifier, self).__init__()
        self.independent_norm       = independent_norm
        self.cls_num        = cls_num
        self.n_prompt       = n_prompts
        if reduce:
            fd = img_dim // 2
            self.icomp      = nn.Sequential(
                nn.BatchNorm1d(img_dim), nn.Conv1d(img_dim, fd, 1), nn.ELU(),
            )
            self.tcomp      = nn.Sequential(
                nn.BatchNorm1d(tex_dim), nn.Conv1d(tex_dim, fd, 1), nn.ELU(),
            )
        else:
            assert tex_dim == img_dim, (
                    'When no use reduce, the image (%d) and text (%d) dim should be same.' % (img_dim, tex_dim)
            )
            self.icomp      = nn.Identity()
            self.tcomp      = nn.Identity()
        self.data_efficient = data_efficient

        in_dim                      = n_prompts * map_num * cls_num
        if self.data_efficient:
            self.fake_cls   = nn.Parameter(torch.randn(self.n_prompt, fake_num, tex_dim), requires_grad=True)
            self.fc         = nn.Sequential(
                nn.LayerNorm(tex_dim),
                nn.Linear(tex_dim, tex_dim, bias=False),
                nn.ELU(),
                nn.Linear(tex_dim, tex_dim, bias=False),
            )
            # cross-part classifier
            in_dim          = n_prompts * map_num * fake_num
        self.fc1 = nn.Sequential(
            nn.LayerNorm(n_prompts * map_num),
            nn.Linear(n_prompts * map_num, n_prompts * map_num, bias=False),
            nn.ELU(),
            nn.Linear(n_prompts * map_num, n_prompts * map_num, bias=False),
        )

        # cross-part classifier
        self.prt_fc                 = nn.Sequential(
            nn.BatchNorm1d(in_dim),
            nn.Linear(in_dim, feat_dim),
            nn.ELU(),
            nn.BatchNorm1d(feat_dim),
            nn.Linear(feat_dim, cls_num, bias=False),
        )

        self.x_sf                   = 64.0
        self.register_buffer('bank', None)

        # self.residual_w     = nn.Parameter(torch.zeros(n_prompts, cls_num, feat_dim))


    def forward(self, img_f, tex_f):
        n, cls, c = tex_f.shape
        # tex_f = tex_f + self.residual_w
        if self.independent_norm:
            img_f = func.normalize(img_f, dim=-1, eps=EPSILON) * self.x_sf
            tex_f = func.normalize(tex_f, dim=-1, eps=EPSILON) * self.x_sf
        else:
            b, n, c = img_f.shape
            img_f = func.normalize(
                img_f.reshape(b, -1), dim=-1, eps=EPSILON
            ).reshape(b, n, c) * self.x_sf
            tex_f = func.normalize(
                tex_f.permute(1, 0, 2).reshape(cls, -1), dim=-1, eps=EPSILON
            ).reshape(cls, n, c).permute(1, 0, 2) * self.x_sf

        ff_size = cls
        if self.data_efficient:
            tmp = torch.einsum(
                'p x c, p y c -> p xy',
                self.fake_cls,   # prompt_num x fake_num x feat_dim
                self.fc(tex_f),  # prompt_num x cls_num  x feat_dim
            )  # prompt_num x fake_num x cls_num
            tex_f = torch.einsum(
                'p xy, p y c -> p x c', tmp, tex_f
            )  # prompt_num x fake_num x feat_dim

            ff_size = self.fake_cls.shape[1]

        cp_rel = torch.einsum(
            'b x c, y d c -> b yxd',
            self.icomp(img_f.permute(0, 2, 1)).permute(0, 2, 1),
            self.tcomp(tex_f.permute(1, 2, 0)).permute(2, 0, 1),
        ).reshape(img_f.shape[0], ff_size, -1)

        proj_cp = self.fc1(cp_rel)
        norm_cp = func.normalize(proj_cp.clone().detach(), dim=-1, eps=EPSILON).mean(0)
        if self.bank is None:
            self.bank = norm_cp
        self.bank = func.normalize(0.1 * norm_cp + 0.9 * self.bank, dim=-1, eps=EPSILON)
        tempp = torch.arange(ff_size).unsqueeze(0).expand(
            img_f.shape[0], -1).to(cp_rel.device).reshape(-1)

        loss = func.cross_entropy(
            torch.einsum(
                'b x c, y c -> b x y',
                func.normalize(proj_cp, dim=-1, eps=EPSILON),
                func.normalize(self.bank, dim=-1, eps=EPSILON),
            ).reshape(-1, ff_size) * 16.0, tempp
        )

        pred = self.prt_fc(cp_rel.reshape(img_f.shape[0], -1))
        return pred, 0.0 * loss
