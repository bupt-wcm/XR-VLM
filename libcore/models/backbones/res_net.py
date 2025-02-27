import torch
import torch.nn as nn
import torchvision.models as tvm
from torch.utils.model_zoo import load_url

def res_back(name, model_type, weights):
    resnet_libs = {
        'resnet': {
            18: tvm.resnet18,
            34: tvm.resnet34,
            50: tvm.resnet50,
            101: tvm.resnet101,
        },
    }
    # the weights of pre-trained resnet is changed, and the performance is different from previous methods
    # a simple test code to align the performance with different SoTAs.
    net = tvm.resnet50()
    state_dict = load_url('https://download.pytorch.org/models/resnet50-19c8e357.pth')
    net.load_state_dict(state_dict)

    # net = tvm.resnet101()
    # state_dict = load_url('https://download.pytorch.org/models/resnet101-5d3b4d8f.pth')
    # net.load_state_dict(state_dict)

    return net
    # return resnet_libs[name][model_type](weights='IMAGENET1K_V2')


class ResBackProxy(nn.Module):
    def __init__(self, name='resnet', model_type=50, weights='IMAGENET1K_V1'):
        super(ResBackProxy, self).__init__()
        backbone = res_back(name, model_type, weights)

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
