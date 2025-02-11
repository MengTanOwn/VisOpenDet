import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class ClassEmbed(nn.Module):
    def __init__(
        self,
        lang_embed_dim: int = 256,
        embed_dim: int = 256,
        return_logit_scare = False
    ):
        super().__init__()
        self.return_logit_scare = return_logit_scare
        self.image_embed_proj = nn.Identity()
        self.lang_embed_proj = nn.Linear(lang_embed_dim, embed_dim, bias=True)
        bias_value = -math.log((1 - 0.01) / 0.01)
        self.lang_bias = nn.ParameterList(
            [nn.Parameter(torch.zeros(lang_embed_dim), requires_grad=True)]
        )
        self.lang_bias0 = nn.ParameterList(
            [nn.Parameter(torch.Tensor([bias_value]), requires_grad=True)]
        )
        self.lang_log_scale = nn.ParameterList(
            [nn.Parameter(torch.Tensor([0.0]), requires_grad=True)]
        )

    def forward(self, image_embeds, lang_embeds):
        lang_embeds = lang_embeds['encoded_support']
        num_queries = image_embeds.shape[1]

        image_embeds = self.image_embed_proj(image_embeds)
        lang_embeds = F.normalize(lang_embeds, p=2, dim=-1)
        lang_embeds_proj = self.lang_embed_proj(lang_embeds / 2.0)
        lang_embeds_bias = (
            torch.einsum("bcd,d->bc", lang_embeds, self.lang_bias[0])
            + self.lang_bias0[0]
        )
        lang_embeds_bias = lang_embeds_bias.unsqueeze(1).repeat(1, num_queries, 1)
        dot_product_logit = (
            torch.einsum("bnd,bcd->bnc", image_embeds, lang_embeds_proj)
            / self.lang_log_scale[0].exp()
        ) + lang_embeds_bias
        dot_product_logit = torch.clamp(dot_product_logit, min=-500, max=500)
        if self.return_logit_scare:
            return dot_product_logit,self.lang_log_scale[0].exp()
        else:
            return dot_product_logit
    

class ContrastiveEmbed(nn.Module):
    def __init__(self, max_support_len=81,norm=False):
        """
        Args:
            max_support_len: max length of support.
        """
        super().__init__()
        self.max_support_len = max_support_len
        self.norm = norm
        if norm:
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        else:
            self.logit_scale = None
        

    def forward(self, x, support_dict):
        """_summary_

        Args:
            x (_type_): _description_
            support_dict (_type_): _description_
            {
                'encoded_support': encoded_support, # bs, 195, d_model
                'support_token_mask': support_token_mask, # bs, 195
                        # True for used tokens. False for padding tokens
            }
        Returns:
            _type_: _description_
        """
        assert isinstance(support_dict, dict)

        y = support_dict["encoded_support"]  #4,13,256  
        support_token_mask = support_dict["support_token_mask"]
        #print(x.shape)
        if self.norm:
            x = F.normalize(x, dim=-1)
            y = F.normalize(y, dim=-1)
            
            logit_scale = self.logit_scale.exp()
            res = logit_scale * x @ y.transpose(-1, -2)
        else:
            res = x @ y.transpose(-1, -2)
        res.masked_fill_(support_token_mask[:, None, :], float("-1e-9"))#-inf

        # padding to max_support_len
        new_res = torch.full((*res.shape[:-1], self.max_support_len), float("-1e-9"), device=res.device)#-inf
        new_res[..., : res.shape[-1]] = res

        return new_res