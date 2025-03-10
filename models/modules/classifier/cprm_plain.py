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
        self.cls_num        = cls_num
        self.n_prompt       = n_prompts
        fd = img_dim // 2
        self.fc1            = nn.Sequential(
            nn.BatchNorm1d(img_dim),
            nn.Conv1d(img_dim, fd, 1), nn.BatchNorm1d(fd), nn.ELU(),
            # nn.Conv1d(fd, fd, 1),
        )
        self.fc2            = nn.Sequential(
            nn.BatchNorm1d(tex_dim),
            nn.Conv1d(tex_dim, fd, 1), nn.BatchNorm1d(fd), nn.ELU(),
            # nn.Conv1d(fd, fd, 1),
        )
        in_dim                      = n_prompts * map_num * cls_num
        # cross-part classifier
        self.prt_fc                 = nn.Sequential(
            nn.BatchNorm1d(in_dim), nn.Linear(in_dim, feat_dim), nn.ELU(),
            nn.BatchNorm1d(feat_dim), nn.Linear(feat_dim, cls_num, bias=False),
        )
        self.x_sf                   = 64.0

    def forward(self, img_f, tex_f):
        n, cls, c = tex_f.shape
        img_f = func.normalize(img_f, dim=-1, eps=EPSILON) * self.x_sf
        tex_f = func.normalize(tex_f, dim=-1, eps=EPSILON) * self.x_sf
        cp_rel = torch.einsum(
            'b x c, y d c -> b yxd',
            self.fc1(img_f.permute(0, 2, 1)).permute(0, 2, 1),
            self.fc2(tex_f.permute(1, 2, 0)).permute(2, 0, 1),
        )

        ##### IMPORTANT ADDED
        # cp_rel = cp_rel.permute(0, 3, 1, 2) # slightly boost the performance, do not mind.
        pred = self.prt_fc(cp_rel.reshape(img_f.shape[0], -1))
        return pred, 0.0 * torch.zeros((1,)).to(pred.device).squeeze()


class XPaReClassifier_cc(nn.Module):
    def __init__(
            self,
            img_dim, tex_dim, feat_dim, n_prompts, cls_num, map_num,
            reduce=False, independent_norm=True,
            fake_num=16, data_efficient=True,
    ):
        super(XPaReClassifier_cc, self).__init__()
        self.cls_num        = cls_num
        self.n_prompt       = n_prompts
        fd = img_dim // 2
        self.fc1            = nn.Sequential(
            nn.BatchNorm1d(img_dim),
            nn.Conv1d(img_dim, fd, 1), nn.BatchNorm1d(fd), nn.ELU(),
            # nn.Conv1d(fd, fd, 1),
        )
        self.fc2            = nn.Sequential(
            nn.BatchNorm1d(tex_dim),
            nn.Conv1d(tex_dim, fd, 1), nn.BatchNorm1d(fd), nn.ELU(),
            # nn.Conv1d(fd, fd, 1),
        )
        in_dim                      = n_prompts * map_num * cls_num
        # cross-part classifier
        self.prt_fc                 = nn.Sequential(
            nn.BatchNorm1d(in_dim), nn.Linear(in_dim, feat_dim), nn.ELU(),
            nn.BatchNorm1d(feat_dim), nn.Linear(feat_dim, cls_num, bias=False),
        )
        self.x_sf                   = 64.0

    def forward(self, img_f, tex_f):
        n, cls, c = tex_f.shape
        img_f = func.normalize(img_f, dim=-1, eps=EPSILON) * self.x_sf
        tex_f = func.normalize(tex_f, dim=-1, eps=EPSILON) * self.x_sf
        cp_rel = torch.einsum(
            'b x c, y d c -> b yxd',
            self.fc1(img_f.permute(0, 2, 1)).permute(0, 2, 1),
            self.fc2(tex_f.permute(1, 2, 0)).permute(2, 0, 1),
        )
        cp_rel = cp_rel.permute(0, 3, 1, 2)
        cp_rel = cp_rel * torch.eye(n, device=tex_f.device).reshape(1, 1, n, n)
        pred = self.prt_fc(cp_rel.reshape(img_f.shape[0], -1))
        return pred #, 0.0 * torch.zeros((1,)).to(pred.device).squeeze()

class XPaReClassifier_cp(nn.Module):
    def __init__(
            self,
            img_dim, tex_dim, feat_dim, n_prompts, cls_num, map_num,
            reduce=False, independent_norm=True,
            fake_num=16, data_efficient=True,
    ):
        super(XPaReClassifier_cp, self).__init__()
        self.cls_num        = cls_num
        self.n_prompt       = n_prompts
        fd = img_dim // 2
        self.fc1            = nn.Sequential(
            nn.BatchNorm1d(img_dim),
            nn.Conv1d(img_dim, fd, 1), nn.BatchNorm1d(fd), nn.ELU(),
            # nn.Conv1d(fd, fd, 1),
        )
        self.fc2            = nn.Sequential(
            nn.BatchNorm1d(tex_dim),
            nn.Conv1d(tex_dim, fd, 1), nn.BatchNorm1d(fd), nn.ELU(),
            # nn.Conv1d(fd, fd, 1),
        )
        in_dim                      = n_prompts * map_num
        # cross-part classifier
        self.prt_fc                 = nn.Sequential(
            nn.BatchNorm1d(in_dim), nn.Linear(in_dim, in_dim // 2), nn.ELU(),
            nn.BatchNorm1d(in_dim // 2), nn.Linear(in_dim // 2, 1, bias=False),
        )
        self.x_sf                   = 64.0

    def forward(self, img_f, tex_f):
        n, cls, c = tex_f.shape
        img_f = func.normalize(img_f, dim=-1, eps=EPSILON) * self.x_sf
        tex_f = func.normalize(tex_f, dim=-1, eps=EPSILON) * self.x_sf
        cp_rel = torch.einsum(
            'b x c, y d c -> b yxd',
            self.fc1(img_f.permute(0, 2, 1)).permute(0, 2, 1),
            self.fc2(tex_f.permute(1, 2, 0)).permute(2, 0, 1),
        )
        cp_rel = cp_rel.permute(0, 3, 1, 2)
        pred = self.prt_fc(cp_rel.reshape(img_f.shape[0] * cls, -1)).reshape(img_f.shape[0], -1)
        return pred, 0.0 * torch.zeros((1,)).to(pred.device).squeeze()


class XPaReClassifier_b(nn.Module):
    def __init__(
            self,
            img_dim, tex_dim, feat_dim, n_prompts, cls_num, map_num,
            reduce=False, independent_norm=True,
            fake_num=16, data_efficient=True,
    ):
        super(XPaReClassifier_b, self).__init__()
        self.cls_num        = cls_num
        self.n_prompt       = n_prompts
        fd = img_dim // 2
        self.fc1            = nn.Sequential(
            nn.BatchNorm1d(img_dim),
            nn.Conv1d(img_dim, fd, 1), nn.BatchNorm1d(fd), nn.ELU(),
            # nn.Conv1d(fd, fd, 1),
        )
        self.fc2            = nn.Sequential(
            nn.BatchNorm1d(tex_dim),
            nn.Conv1d(tex_dim, fd, 1), nn.BatchNorm1d(fd), nn.ELU(),
            # nn.Conv1d(fd, fd, 1),
        )
        in_dim                      = n_prompts * map_num
        # cross-part classifier
        self.prt_fc                 = nn.Sequential(
            nn.BatchNorm1d(in_dim), nn.Linear(in_dim, in_dim // 2), nn.ELU(),
            nn.BatchNorm1d(in_dim // 2), nn.Linear(in_dim // 2, 1, bias=False),
        )
        self.x_sf                   = 64.0

    def forward(self, img_f, tex_f):
        n, cls, c = tex_f.shape
        img_f = func.normalize(img_f, dim=-1, eps=EPSILON) * self.x_sf
        tex_f = func.normalize(tex_f, dim=-1, eps=EPSILON) * self.x_sf
        cp_rel = torch.einsum(
            'b x c, y d c -> b yxd',
            self.fc1(img_f.permute(0, 2, 1)).permute(0, 2, 1),
            self.fc2(tex_f.permute(1, 2, 0)).permute(2, 0, 1),
        ) # batch x prompt_num x map_num x cls_num
        # print(img_f.shape, tex_f.shape, cp_rel.shape)
        cp_rel = cp_rel.permute(0, 3, 1, 2)
        cp_rel = cp_rel * torch.eye(n, device=tex_f.device).reshape(1, 1, n, n)

        pred = self.prt_fc(cp_rel.reshape(img_f.shape[0] * cls, -1)).reshape(img_f.shape[0], -1)
        return pred, 0.0 * torch.zeros((1,)).to(pred.device).squeeze()