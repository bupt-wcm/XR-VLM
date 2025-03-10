import torch
import torch.nn as nn
import torch.nn.functional as func


class UniAttPooling(nn.Module):
    def __init__(self, in_dim, map_num, softmax_scale):
        super(UniAttPooling, self).__init__()
        self.map_num = map_num
        self.softmax_scale = softmax_scale
        self.atte = nn.Sequential(
            nn.BatchNorm1d(in_dim),
            nn.Conv1d(in_dim, map_num + 1, 1, 1, 0, bias=False),
        )
        self.post_bn = nn.Sequential(
            nn.BatchNorm1d(map_num + 1),
            nn.ReLU(),
        )

    def spatial(self, x):
        spatial_maps = self.atte(x)
        if self.softmax_scale > 0.0:
            spatial_maps = torch.softmax(spatial_maps * self.softmax_scale, dim=-1)
        spatial_maps = self.post_bn(spatial_maps)
        return spatial_maps

    def pooling(self, spatial_maps, feat):
        b, n, d = spatial_maps.shape
        ff = torch.einsum('b m n, b c n -> b m c', spatial_maps, feat)
        ff = torch.sign(ff) * torch.sqrt(torch.add(torch.abs(ff), 1e-6))
        ff = func.normalize(ff.reshape(b, -1), dim=-1, p=2).reshape(b, n, -1)
        return ff[:, :self.map_num, :]  # remove the background maps from the features.

    def forward(self, x, f):
        assert x.dim() == 3, (
                'The input dimension of uni attention module should be (B, C, N), '
                'but current shape is %s' % x.dim())
        spatial_maps = self.spatial(x)
        f = self.pooling(spatial_maps, f)
        return f
