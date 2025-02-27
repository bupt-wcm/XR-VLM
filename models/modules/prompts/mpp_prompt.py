import torch
import torch.nn as nn
from libcore.models.clip import clip
from libcore.models.clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from models.data_name import name_dict

_tokenizer = _Tokenizer()


class MppPrompt(nn.Module):
    def __init__(self, clip_model, data_name, params, ):
        super(MppPrompt, self).__init__()

        classnames, _   = name_dict[data_name]
        dtype           = clip_model.dtype
        ctx_dim         = clip_model.ln_final.weight.shape[0]
        clip_imsize     = clip_model.visual.input_resolution
        cfg_imsize      = params.IMG_SIZE
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        self.n_cls      = len(classnames)
        self.n_fix      = params.N_FIX      # prompts shared by different classes
        self.n_fle      = params.N_FLE      # prompts specified for different classes
        self.n_split    = params.N_SPLIT    # how many (n_fix + n_fle) for each group
        self.n_group    = params.N_GROUP    # `map_num` for prompts
        self.alternate  = params.ALTERNATE  # the order of n_fix and n_fle
        self.n_ctx      = (self.n_fix + self.n_fle) * self.n_split

        # random initialization
        fle_vectors     = torch.empty(self.n_group, self.n_cls, self.n_split, self.n_fle, ctx_dim, dtype=dtype)
        fix_vectors     = torch.empty(self.n_group, 1,          self.n_split, self.n_fix, ctx_dim, dtype=dtype)
        nn.init.normal_(fle_vectors, std=0.02)
        nn.init.normal_(fix_vectors, std=0.02)
        self.fle_ctx    = nn.Parameter(fle_vectors)  # to be optimized
        self.fix_ctx    = nn.Parameter(fix_vectors)  # to be optimized

        classnames      = [name.replace("_", " ") for name in classnames]
        name_lens       = [len(_tokenizer.encode(name)) for name in classnames]

        prompts         = [" ".join(["X"] * self.n_ctx) + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding   = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + self.n_ctx:, :])  # CLS, EOS
        self.tokenized_prompts = \
            tokenized_prompts.unsqueeze(0).expand(self.n_group, -1, -1).reshape(self.n_group * len(classnames), -1)
        self.name_lens = name_lens
        self.class_token_position = params.CLASS_TOKEN_POSITION

    def forward(self):

        prefix, suffix = self.token_prefix, self.token_suffix

        list_prompts = []
        for g_idx in range(self.n_group):
            fle_vec, fix_vec = self.fle_ctx[g_idx], self.fix_ctx[g_idx]
            if self.alternate:
                assert min(fix_vec.shape[-2], fle_vec.shape[-2]) != 0, \
                    'alternate only support when size of fix_vec and fle_vec is not 0.'
                fix_vec = fix_vec.expand(self.n_cls, -1, -1, -1)  # expand the single dimension to class number.
                ctx = [sub[idx] for idx in range(len(fle_vec)) for sub in [fix_vec, fle_vec,]]
                ctx = torch.cat(ctx, dim=1).reshape(self.n_cls, self.n_ctx, -1)
            else:
                if fix_vec.shape[-2] == 0:
                    ctx = fle_vec.reshape(self.n_cls, self.n_split * self.n_fle, -1)
                elif fle_vec.shape[-2] == 0:
                    fix_vec = fix_vec.expand(self.n_cls, -1, -1, -1)
                    ctx = fix_vec.reshape(self.n_cls, self.n_split * self.n_fix, -1)
                else:
                    fix_vec = fix_vec.expand(self.n_cls, -1, -1, -1)
                    ctx = torch.cat([
                        fix_vec.reshape(self.n_cls, self.n_split * self.n_fix, -1),
                        fle_vec.reshape(self.n_cls, self.n_split * self.n_fle, -1),
                    ], dim=1)

            prompts = torch.cat(
                [
                    prefix,     # (n_cls, 1,     dim)
                    ctx,        # (n_cls, n_ctx, dim)
                    suffix,     # (n_cls, *,     dim)
                ],
                dim=1,
            )
            list_prompts.append(prompts)
        return torch.stack(list_prompts, dim=0)  # part_num x num_cls x text_len x embed_dim