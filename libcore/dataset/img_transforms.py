import torchvision.transforms as torchtfs


class DefaultTransform:
    def __init__(self, input_cfg):
        scale_size, crop_size = input_cfg.SIZE_SCALE, input_cfg.SIZE_IMAGE

        if len(scale_size) == 2:
            resize_func = torchtfs.Resize((scale_size[0], scale_size[1]))
        elif len(scale_size) == 1:
            resize_func = torchtfs.Resize(scale_size)
        else:
            raise KeyError('The length of scale size (tuple type) should be 1 or 2.')

        self.train_trans = torchtfs.Compose(
            [
                resize_func,
                torchtfs.RandomRotation(15),
                torchtfs.RandomCrop((crop_size, crop_size), padding=8),
                torchtfs.RandomHorizontalFlip(),
                torchtfs.ToTensor(),
                torchtfs.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

        self.valid_trans = torchtfs.Compose(
            [
                torchtfs.Resize(scale_size),
                torchtfs.CenterCrop((crop_size, crop_size)),
                torchtfs.ToTensor(),
                torchtfs.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )


class CoOpTransform:
    def __init__(self, input_cfg):
        scale_size, crop_size = input_cfg.SIZE_SCALE, input_cfg.SIZE_IMAGE
        self.train_trans = torchtfs.Compose(
            [
                torchtfs.RandomResizedCrop(crop_size),
                torchtfs.RandomHorizontalFlip(),
                torchtfs.ToTensor(),
                torchtfs.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ]
        )

        self.valid_trans = torchtfs.Compose(
            [
                torchtfs.Resize(max(scale_size)),
                torchtfs.CenterCrop(crop_size),
                torchtfs.ToTensor(),
                torchtfs.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ]
        )


def build_transform(input_cfg):
    trans_name = input_cfg.TRANS_NAME
    if trans_name == 'DEFAULT' or trans_name is None:
        return DefaultTransform(input_cfg)
    elif trans_name == 'CoOp':
        return CoOpTransform(input_cfg)
    else:
        raise NotImplementedError
