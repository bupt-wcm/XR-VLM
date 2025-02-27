from .res_net import ResBackProxy


def build_backbone(name, model_type):
    if name in ['resnet', 'resnext']:
        return ResBackProxy(name, model_type=model_type)
    else:
        raise KeyError('Unimplemented Backbone Network %s' % name)
