import torch
import torch.nn as nn
import torch.nn.functional as func

class MpConLossModule(nn.Module):
    def __init__(self, in_dim1, in_dim2, ):
        super(MpConLossModule, self).__init__()
        self.fc_1 = nn.Sequential(
            nn.Linear(in_dim1, in_dim1 // 2, bias=False),
            nn.BatchNorm1d(in_dim1 // 2), nn.ReLU(), nn.Linear(in_dim1 // 2, in_dim2, bias=False),
        )
        self.fc_2 = nn.Sequential(
            nn.Linear(in_dim2, in_dim2 // 2, bias=False),
            nn.BatchNorm1d(in_dim2 // 2), nn.ReLU(), nn.Linear(in_dim2 // 2, in_dim1, bias=False),
        )


    def forward(self, img_f, tex_f, lb):
        tex_f = tex_f[:, lb].permute(1, 0, 2)
        b, n, c = img_f.shape
        m, p, c = tex_f.shape
        if n == 1 or m == 1 or n != m:
            return torch.zeros(size=(1, ), device=img_f.device).squeeze()
        p1 = func.normalize(self.fc_1(img_f.reshape(-1, c)), dim=-1).reshape(b, n, -1) * 64.
        p2 = func.normalize(self.fc_2(tex_f.reshape(-1, c)), dim=-1).reshape(m, p, -1) * 64.

        fk_tf = func.normalize(tex_f, dim=-1, p=2) * 64.
        fk_if = func.normalize(img_f, dim=-1, p=1) * 64.
        ld_lb = torch.arange(n, device=img_f.device).unsqueeze(0).expand(b, -1).reshape(-1)
        loss = func.cross_entropy(
            torch.einsum('b n c, b m c -> b n m', p1, fk_tf.detach()).reshape(b*n, n), ld_lb
        ) + func.cross_entropy(
            torch.einsum('b n c, b m c -> b n m', p2, fk_if.detach()).reshape(b*n, n), ld_lb
        )
        return loss * 0.5
