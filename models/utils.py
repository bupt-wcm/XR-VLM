import os
import math
import torch
import torch.nn as nn
from libcore.models.clip import clip


def load_clip_to_cpu(bb_cfg):
    model_name, image_size, stride_size = bb_cfg.NAME, bb_cfg.IMAGE_SIZE, bb_cfg.STRIDE_SIZE
    url = clip._MODELS[bb_cfg.NAME]
    model_path = clip._download(url, root=os.path.expanduser("~/.cache/clip"))

    if isinstance(image_size, tuple):
        image_size = image_size[0]

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = model.state_dict()
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    if 'vit' in model_name.lower():
        # change the pos embedding in visual encoder
        pos_embedding = state_dict["visual.positional_embedding"]
        pos_embedding = pos_embedding.unsqueeze(0)  # (1, seq_length, hidden_dim)
        n, seq_length, hidden_dim = pos_embedding.shape
        patch_size = state_dict["visual.conv1.weight"].shape[-1]

        if stride_size is None or stride_size == patch_size:
            new_seq_length = (image_size // patch_size) ** 2 + 1
        else:
            new_seq_length = ((image_size - patch_size) // stride_size + 1) ** 2 + 1

        seq_length -= 1
        new_seq_length -= 1
        pos_embedding_cls = pos_embedding[:, :1, :]
        pos_embedding_img = pos_embedding[:, 1:, :]
        if seq_length != new_seq_length:
            print('Setting the Seq length from %d to %d.' % (seq_length, new_seq_length))

        # (1, seq_length, hidden_dim) -> (1, hidden_dim, seq_length)
        pos_embedding_img = pos_embedding_img.permute(0, 2, 1)
        seq_length_1d = int(math.sqrt(seq_length))
        pos_embedding_img = pos_embedding_img.reshape(1, hidden_dim, seq_length_1d, seq_length_1d)
        new_seq_length_1d = int(math.sqrt(new_seq_length))
        new_pos_embedding_img = nn.functional.interpolate(
            pos_embedding_img,
            size=new_seq_length_1d,
            mode='bicubic',
            align_corners=True,
        )

        # (1, hidden_dim, new_seq_l_1d, new_seq_l_1d) -> (1, hidden_dim, new_seq_length)
        new_pos_embedding_img = new_pos_embedding_img.reshape(1, hidden_dim, new_seq_length)

        # (1, hidden_dim, new_seq_length) -> (1, new_seq_length, hidden_dim)
        new_pos_embedding_img = new_pos_embedding_img.permute(0, 2, 1)
        new_pos_embedding = torch.cat([pos_embedding_cls, new_pos_embedding_img], dim=1)

        state_dict["visual.positional_embedding"] = new_pos_embedding.squeeze(0)
        state_dict["visual.stride_size"] = stride_size

    model = clip.build_model(state_dict)
    if 'rn' in model_name.lower():
        assert bb_cfg.LAST_STRIDE in [1, 2, 4], \
            'Setting stride size of the RN should be 1(keep same), 2(upsample 2x), 4(upsample 4x)'
        if bb_cfg.LAST_STRIDE > 1:
            for m in model.visual.layer3.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.AvgPool2d):
                    m.stride = 1
        if bb_cfg.LAST_STRIDE > 2:
            for m in model.visual.layer4.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.AvgPool2d):
                    m.stride = 1

    model.visual.input_resolution = image_size

    return model

def make_pair(x):
    if isinstance(x, tuple) or isinstance(x, list):
        assert len(x) == 2, len(x)
        return x
    else:
        return x, x
