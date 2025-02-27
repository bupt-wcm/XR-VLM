import torch
import torch.nn as nn


class TexEncoder(nn.Module):
    def __init__(self, clip_model):
        super(TexEncoder, self).__init__()
        self.transformer            = clip_model.transformer
        self.ln_final               = clip_model.ln_final
        self.text_projection        = clip_model.text_projection
        self.dtype                  = clip_model.dtype
        self.positional_embedding   = clip_model.positional_embedding

    def forward(self, x, tokenized_prompts):
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x
