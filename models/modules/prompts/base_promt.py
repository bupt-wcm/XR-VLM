import torch
import torch.nn as nn
from models.data_name import name_dict
from libcore.models.clip import clip


class BasePrompts(nn.Module):
    """
    no learnable prompts in BasePrompts class, only use some handcraft prompts to present the
    - zero-shot performance
    - fine-tune performance of visual encoder
    - the differences of different handcraft prompts
    """
    def __init__(self, clip_model, data_name):
        super(BasePrompts, self).__init__()
        self.cls_name, self.temp = name_dict[data_name]

        tokenized_prompts = torch.cat([clip.tokenize(self.temp[0].format(cls_name=p)) for p in self.cls_name])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts)
        # self.prompts = embedding
        # self.tokenized_prompts = tokenized_prompts

        self.register_buffer('prompts', embedding)
        self.register_buffer('tokenized_prompts', tokenized_prompts)

    def forward(self, ):
        return self.prompts
