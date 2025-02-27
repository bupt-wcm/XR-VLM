import torch
import torch.nn as nn

class ImPaClassifier(nn.Module):  # Image Part Feature Classifier
    def __init__(self, img_dim, feat_dim, cls_num, map_num):
        super(ImPaClassifier, self).__init__()
        self.mp_mlp = nn.ModuleList(
            [
                nn.Sequential(
                    nn.BatchNorm1d(img_dim),
                    nn.Linear(img_dim, feat_dim),
                    nn.ELU(),
                    nn.BatchNorm1d(feat_dim),
                    nn.Linear(feat_dim, cls_num),
                ) for _ in range(map_num)
            ]
        )
        self.ip_sf = 64.0
    def forward(self, img_f):
        b, n, c = img_f.shape
        img_pred = torch.stack(
            [self.mp_mlp[i](img_f[:, i] * self.ip_sf) for i in range(n)], dim=0
        ).mean(dim=0, keepdim=False)
        return img_pred
