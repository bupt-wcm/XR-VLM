import torch
import torch.nn as nn
import torchvision.models as tvm


def vit_back(name, model_type, weights):
    vit_libs = {
        'vit_big': {
            16: tvm.vit_b_16,
            32: tvm.vit_b_32,
        },

    }
    return vit_libs[name][model_type](weights=weights)


class VitBackProxy(nn.Module):
    def __init__(self, name='resnet', model_type=50, weights='IMAGENET1K_V1'):
        super(VitBackProxy, self).__init__()
        backbone = vit_back(name, model_type, weights)

        self.pre_conv = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )

        layers = []
        for i in range(4):
            layers.append(getattr(backbone, 'layer' + str(i + 1)))
        self.layers = nn.ModuleList(layers)
        self.max_output_num = 4

        self.out_dims = []
        x = self.pre_conv(torch.randn(1, 3, 128, 128))
        for layer in self.layers:
            x = layer(x)
            self.out_dims.append(x.shape[1])
        del x
        del backbone

    def forward(self, inp, out_ids=None):
        x = self.pre_conv(inp)
        if out_ids is None:
            for idx, layer in enumerate(self.layers):
                x = layer(x)
            return x
        else:
            xs = []
            for idx, layer in enumerate(self.layers):
                x = layer(x)
                if idx in out_ids:
                    xs.append(x)
            return xs
