import os
import json
import torch
import torch.nn as nn
from easydict import EasyDict as edict
from libcore.models.clip import clip
from libcore.models.clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from models.data_name import name_dict

# hand-craft prompts from LLM

_tokenizer = _Tokenizer()


PART_DES_PATH = './data/part_des'
FILE_FORMAT = '{data_name}_{part_num}_des_dataset.json'


class HcPrompt(nn.Module):
    def __init__(self, clip_model, data_name, params):
        super(HcPrompt, self).__init__()

        file_path = FILE_FORMAT.format(data_name=data_name, part_num=params.N_GROUP)
        with open(os.path.join(PART_DES_PATH, file_path), 'r') as f:
            des_dict = edict(json.load(f))

        self.cls_name, self.temp = name_dict[data_name]
        part_keys = None
        part_des = []
        for c in self.cls_name:
            cls_des = des_dict[c]
            if part_keys is None:
                part_keys = list(cls_des.keys())
            for p in part_keys:
                part_des.append(cls_des[p])
        self.cls_num, self.prt_num = len(des_dict.keys()), len(part_keys)
        self.n_group = self.prt_num
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in part_des])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts)
        # self.prompts = embedding
        # self.tokenized_prompts = tokenized_prompts

        self.register_buffer('prompts', embedding)
        self.register_buffer('tokenized_prompts', tokenized_prompts)

    def forward(self, ):
        # part_num x num_cls x text_len x embed_dim
        _, tex_len, emb_dim = self.prompts.shape
        return self.prompts.reshape(self.cls_num, self.prt_num, tex_len, emb_dim).permute(1, 0, 2, 3)
