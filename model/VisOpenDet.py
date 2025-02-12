import numpy as np 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 

from .utils import NestedTensor
from.positional_encoding import PositionEmbeddingSineHW


class VisOpenDet(nn.Module):
    def __init__(self, backbone: nn.Module, encoder, decoder, multi_scale=None,l2_norm=False,backbone_num_channels=[512, 1024, 2048],num_feature_levels=4,freeze_encoder = False):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder
        self.multi_scale = multi_scale
        self.l2_norm = l2_norm
        self.backbone_num_channels = backbone_num_channels[4-num_feature_levels:]
        self.num_feature_levels = num_feature_levels
        self.freeze_encoder = freeze_encoder
        self.hidden_dim = 256 
        if self.freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False        
    def forward(self, x,y=None,targets=None,text_features=None,vp=None,extract_feature_mode=False,cross_vp=False):
        x_mask = x.mask
        if y!=None:
            y_mask = y.mask
            x,y = x.tensors,y.tensors
        else:
            x = x.tensors
        if self.multi_scale and self.training:
            sz = np.random.choice(self.multi_scale)
            x = F.interpolate(x, size=[sz, sz])
            x_mask = F.interpolate(x_mask, size=[sz, sz])
        if hasattr(self.backbone,'get_intermediate_layers'):
            with torch.no_grad():
                x =self.backbone.get_intermediate_layers(x,reshape=True)
        else:
            if hasattr(self.backbone,'fusion_1'):
                x = self.backbone(x,y)
            else:
                x = self.backbone(x)
        x = [feat.permute(0, 3, 1, 2).contiguous() for feat in x]
        x,prompt_dict = self.encoder(x,vp=vp,query_mask=x_mask)
        x = self.decoder(x,prompt_dict,targets)    
        return x
    
    def deploy(self,):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self 
    def extend_featlayer(self, features, pos):
        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])

                # m = samples.mask
                # mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                # todo
                mask = F.interpolate(mask[None].float(), size=src.shape[-2:]).to(
                    torch.bool
                )[0]

                # pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                # todo
                pos_l = self.pos_emb(NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        return srcs, masks, pos
    def __build_input_proj__(self):
        num_backbone_outs = len(self.backbone_num_channels)
        input_proj_list = []
        for _ in range(num_backbone_outs):
            in_channels = self.backbone_num_channels[_]
            input_proj_list.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, self.hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, self.hidden_dim),
                )
            )
        for _ in range(self.num_feature_levels - num_backbone_outs):
            input_proj_list.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels, self.hidden_dim, kernel_size=3, stride=2, padding=1
                    ),
                    nn.GroupNorm(32, self.hidden_dim),
                )
            )
            in_channels = self.hidden_dim

        return nn.ModuleList(input_proj_list)
    def build_position_encoding(args):
        position_embedding = PositionEmbeddingSineHW(
            128,
            temperatureH=20,
            temperatureW=20,
            normalize=True,
        )
        return position_embedding
    
