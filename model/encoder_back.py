'''by lyuwenyu
'''

import copy
import torch 
from torch import Tensor
import torch.nn as nn 
import torch.nn.functional as F 
from .BMHA import BiAttentionBlock
from .decoder import MSDeformableAttention as MSDeformAttn
from .prompt_encoder import PositionEmbeddingRandom
from .utils import get_activation,_get_clones,get_sine_pos_embed,_get_activation_fn
import random
class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding=None, bias=False, act=None):
        super().__init__()
        self.conv = nn.Conv2d(
            ch_in, 
            ch_out, 
            kernel_size, 
            stride, 
            padding=(kernel_size-1)//2 if padding is None else padding, 
            bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act) 

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class RepVggBlock(nn.Module):
    def __init__(self, ch_in, ch_out, act='relu'):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv1 = ConvNormLayer(ch_in, ch_out, 3, 1, padding=1, act=None)
        self.conv2 = ConvNormLayer(ch_in, ch_out, 1, 1, padding=0, act=None)
        self.act = nn.Identity() if act is None else get_activation(act) 

    def forward(self, x):
        if hasattr(self, 'conv'):
            y = self.conv(x)
        else:
            y = self.conv1(x) + self.conv2(x)

        return self.act(y)

    def convert_to_deploy(self):
        if not hasattr(self, 'conv'):
            self.conv = nn.Conv2d(self.ch_in, self.ch_out, 3, 1, padding=1)

        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv.weight.data = kernel
        self.conv.bias.data = bias 
        # self.__delattr__('conv1')
        # self.__delattr__('conv2')

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1), bias3x3 + bias1x1

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return F.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch: ConvNormLayer):
        if branch is None:
            return 0, 0
        kernel = branch.conv.weight
        running_mean = branch.norm.running_mean
        running_var = branch.norm.running_var
        gamma = branch.norm.weight
        beta = branch.norm.bias
        eps = branch.norm.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


class CSPRepLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks=3,
                 expansion=1.0,
                 bias=None,
                 act="silu"):
        super(CSPRepLayer, self).__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.conv2 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.bottlenecks = nn.Sequential(*[
            RepVggBlock(hidden_channels, hidden_channels, act=act) for _ in range(num_blocks)
        ])
        if hidden_channels != out_channels:
            self.conv3 = ConvNormLayer(hidden_channels, out_channels, 1, 1, bias=bias, act=act)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x):
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        x_2 = self.conv2(x)
        return self.conv3(x_1 + x_2)


# transformer
class TransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu",
                 normalize_before=False):
        super().__init__()
        self.normalize_before = normalize_before

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = get_activation(activation) 

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        residual = src
        if self.normalize_before:
            src = self.norm1(src)
        q = k = self.with_pos_embed(src, pos_embed)
        src, _ = self.self_attn(q, k, value=src, attn_mask=src_mask)

        src = residual + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1(src)

        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)
        if not self.normalize_before:
            src = self.norm2(src)
        return src


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=src_mask, pos_embed=pos_embed)

        if self.norm is not None:
            output = self.norm(output)

        return output



class HybridEncoder(nn.Module):
    def __init__(self,
                 in_channels=[512, 1024, 2048],
                 raw_support_feat_dim = 384,
                 feat_strides=[8, 16, 32],
                 hidden_dim=256,
                 nhead=8,
                 dim_feedforward = 1024,
                 dropout=0.0,
                 enc_act='gelu',
                 use_encoder_idx=[2],
                 num_encoder_layers=1,
                 pe_temperature=10000,
                 expansion=1.0,
                 depth_mult=1.0,
                 act='silu',
                 BMHA=False,
                 num_fusion_layers = 2,
                 use_mask_head = False,
                 eval_spatial_size=None,
                 use_text = False,
                 use_visual_prompt = False,
                 max_support_len = 100,
                 ):
        super().__init__()
        self.max_support_len = max_support_len
        self.in_channels = in_channels
        self.use_text = use_text
        self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = use_encoder_idx
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.eval_spatial_size = eval_spatial_size
        self.BMHA = BMHA
        self.use_mask_head = use_mask_head
        # if use_visual_prompt:
        #     self.pe_layer = PositionEmbeddingRandom(hidden_dim // 2)
        #     self.num_point_embeddings: int = 4  # pos/neg point + 2 box corners
        #     point_embeddings = [nn.Embedding(1, hidden_dim) for i in range(self.num_point_embeddings)]
        #     self.point_embeddings = nn.ModuleList(point_embeddings)
        if use_visual_prompt:
            self.pe_layer = PositionEmbeddingRandom(hidden_dim // 2)
            self.num_point_embeddings: int = 4  # pos/neg point + 2 box corners
            point_embeddings = [nn.Embedding(1, hidden_dim) for i in range(self.num_point_embeddings)]
            self.point_embeddings = nn.ModuleList(point_embeddings)
            self.content_embedding = nn.Embedding(1, hidden_dim)
            self.use_visual_prompt = use_visual_prompt
            self.cross_attention_vp = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout)
            self.cross_attention_vp_dropout = (
                nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            )
            self.cross_attention_vp_norm = nn.LayerNorm(hidden_dim)
            self.self_attn = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout,batch_first=True)
            self.dropout_post = nn.Dropout(dropout)
            self.norm_post = nn.LayerNorm(hidden_dim)
        if self.BMHA:
            self.fusion_layers = nn.ModuleList()
            for _ in range(num_fusion_layers):
                self.fusion_layers.append(
                    copy.deepcopy(BiAttentionBlock(
                    v_dim=hidden_dim,
                    l_dim=hidden_dim,
                    embed_dim=dim_feedforward // 2,
                    num_heads=nhead // 2,
                    dropout=dropout,
            ))
        )



        self.out_channels = [hidden_dim for _ in range(len(in_channels))]
        self.out_strides = feat_strides
        
        # channel projection
        self.input_proj = nn.ModuleList()
        for in_channel in in_channels:
            self.input_proj.append(
                nn.Sequential(
                    nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(hidden_dim)
                )
            )
        self.support_input_proj = nn.Linear(raw_support_feat_dim, hidden_dim)
        self.text_input_proj = nn.Linear(512, hidden_dim)

        # encoder transformer
        encoder_layer = TransformerEncoderLayer(
            hidden_dim, 
            nhead=nhead,
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            activation=enc_act)

        self.encoder = nn.ModuleList([
            TransformerEncoder(copy.deepcopy(encoder_layer), num_encoder_layers) for _ in range(len(use_encoder_idx))
        ])

        # top-down fpn
        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1, 0, -1):
            self.lateral_convs.append(ConvNormLayer(hidden_dim, hidden_dim, 1, 1, act=act))
            self.fpn_blocks.append(
                CSPRepLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion)
            )

        # bottom-up pan
        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1):
            self.downsample_convs.append(
                ConvNormLayer(hidden_dim, hidden_dim, 3, 2, act=act)
            )
            self.pan_blocks.append(
                CSPRepLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion)
            )

        self._reset_parameters()
    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos
    def _embed_boxes(self, boxes: torch.Tensor,input_image_size) -> torch.Tensor:
        """Embeds box prompts."""
        boxes = boxes + 0.5  # Shift to center of pixel
        coords = boxes.reshape(-1, 2, 2)
        corner_embedding = self.pe_layer.forward_with_coords(coords,input_image_size)
        corner_embedding[:, 0, :] += self.point_embeddings[2].weight
        corner_embedding[:, 1, :] += self.point_embeddings[3].weight
        #average the two corners
        corner_embedding = corner_embedding.mean(dim=1)
        return corner_embedding
    def _reset_parameters(self):
        if self.eval_spatial_size:
            for idx in self.use_encoder_idx:
                stride = self.feat_strides[idx]
                pos_embed = self.build_2d_sincos_position_embedding(
                    self.eval_spatial_size[1] // stride, self.eval_spatial_size[0] // stride,
                    self.hidden_dim, self.pe_temperature)
                setattr(self, f'pos_embed{idx}', pos_embed)
                # self.register_buffer(f'pos_embed{idx}', pos_embed)

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.):
        '''
        '''
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
        assert embed_dim % 4 == 0, \
            'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature ** omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.concat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)[None, :, :]

    def forward(self, feats,support_feat=None,query_mask=None,support_mask=None,text_features=None,vp=None,cross_vp=False):
        assert len(feats) == len(self.in_channels)
        query_mask_org = query_mask
        if support_feat!=None:
            support_feat = support_feat.type_as(feats[0])
            if cross_vp == False:
                # support_feat = support_feat.transpose(1,2)
                support_feat = self.support_input_proj(support_feat) #bs num_cls 256
            else:
                support_feat = support_feat.transpose(1,2)
                # support_feat = self.support_input_proj(support_feat)

            
            if not self.training and text_features is not None:
                text_features.tensors[:,:,0] = torch.zeros_like(text_features.tensors[:,:,0])
            if self.use_text:
                text_feat = self.text_input_proj(text_features.tensors.transpose(1,2).type_as(support_feat)) if text_features is not None else 0
                if self.training and text_features is not None:
                    support_feat += text_feat if random.random() > 0.5 else 0       
        elif vp!=None:
            bs = feats[0].shape[0]
            support_feat = torch.zeros((bs,self.max_support_len, self.hidden_dim), device=feats[0].device)
            #sparse_embeddings = torch.empty((bs, 0, self.hidden_dim), device=feats[0].device)
            box_embeddings = [self._embed_boxes(each['boxes'],each['input_image_size']) for each in vp]
            #pad box_embeddings to the same length
            max_len = max([len(each['boxes'][0]) for each in vp])
            for i in range(len(box_embeddings)):
                box_embeddings[i] = F.pad(box_embeddings[i], (0, 0, 0, max_len - len(box_embeddings[i])))
            #get support mask based on the length of box_embeddings
            support_mask = torch.zeros((bs, self.max_support_len), device=feats[0].device).type_as(query_mask)    
            box_embeddings = torch.stack(box_embeddings,dim=0)
            corresponding_cates = [each['cates'] for each in vp]
            #sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=1)
            #average box_embeddings that have the same category
            # box_embeddings = self.support_input_proj(box_embeddings)
            for i in range(bs):
                unique_cates = torch.unique(corresponding_cates[i])
                for cate in unique_cates:
                    mask = corresponding_cates[i] == cate
                    #pad mask
                    mask = F.pad(mask, (0, max_len - len(mask)))
                    try:
                        support_feat[i, cate] = box_embeddings[i, mask].mean(dim=0)
                    except:
                        import pdb;pdb.set_trace()
            # 生成一个随机张量,
            # random_tensor = torch.randn((bs, self.max_support_len, self.hidden_dim), device=feats[0].device)
            # support_feat += random_tensor#在线性映射之前添加随机扰动
            support_feat = self.support_input_proj(support_feat)#存在BUG，无法进行跨图推理指标计算。
            # support_feat += random_tensor#在线性映射之后添加随机扰动TODO
            
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        # if self.BMHA:
        #     for i in range(len(proj_feats)):
        #         h, w = proj_feats[i].shape[2:]
        #         proj_mask = F.interpolate(query_mask[None].float(), size=(h,w)).to(torch.bool)[0]
        #         proj_mask_flatten = proj_mask.flatten(1)
        #         proj_feat_flatten = proj_feats[i].flatten(2).permute(0, 2, 1)
        #         proj_feat_flatten,support_feat = self.BiAttn(proj_feat_flatten,support_feat,proj_mask_flatten,support_mask)
        #         proj_feats[i] = proj_feat_flatten.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w).contiguous()
        if len(proj_feats) == 1:
            self.use_encoder_idx = [0]
        # encoder
        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                h, w = proj_feats[enc_ind].shape[2:]
                query_mask = F.interpolate(query_mask[None].float(), size=(h,w)).to(torch.bool)[0]
                # flatten [B, C, H, W] to [B, HxW, C]
                src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)
                query_mask_flatten = query_mask.flatten(1)
                if self.training or self.eval_spatial_size is None:
                    pos_embed = self.build_2d_sincos_position_embedding(
                        w, h, self.hidden_dim, self.pe_temperature).to(src_flatten.device)
                else:
                    pos_embed = getattr(self, f'pos_embed{enc_ind}', None).to(src_flatten.device)#1,hw, 256
                
                #concat with query image features
                org_shape = src_flatten.shape
                memory = self.encoder[i](src_flatten, pos_embed=pos_embed)
                if self.BMHA and cross_vp == False:
                    #query_mask = F.interpolate(query_mask[None].float(), size=src_flatten.shape[-2:]).to(torch.bool)[0]
                    #src_flatten,support_feat = self.BiAttn(src_flatten,support_feat,query_mask_flatten,support_mask)
                    for index in range(len(self.fusion_layers)):
                        _,support_feat = self.fusion_layers[index](src_flatten,support_feat,query_mask_flatten,support_mask)
                # if cross_vp == False:
                #     src_flatten = torch.cat([src_flatten,support_feat],dim=1)
                #     pos_embed = torch.cat([pos_embed,torch.zeros_like(support_feat[-1:,:,:])],dim=1)
                #     memory = self.encoder[i](src_flatten, pos_embed=pos_embed)
                #     support_feat = memory[:,org_shape[1]:,:]
                # else:
                
                
                memory = F.interpolate(memory.transpose(1,2),size=org_shape[1]).transpose(1,2)#wuyong
                if self.use_visual_prompt and cross_vp == False:
                    Q = self.content_embedding.weight[None, :]
                    #expand to the same size as support_feat
                    Q = Q.expand(support_feat.shape[0], support_feat.shape[1], support_feat.shape[2])#可能存在问题，所有类别公用同一个权重？
                    Q_ = self.cross_attention_vp(self.with_pos_embed(Q.transpose(0,1),support_feat.transpose(0,1)), memory.transpose(0,1), memory.transpose(0,1),query_mask_flatten)[0].transpose(0,1)
                    Q = Q + self.cross_attention_vp_dropout(Q_)
                    Q = self.cross_attention_vp_norm(Q)
                    q = k = self.with_pos_embed(Q, support_feat)
                    Q_, _ = self.self_attn(q, k, value=Q, attn_mask=None)
                    Q = Q + self.dropout_post(Q_)
                    support_feat = self.norm_post(Q)
                
                proj_feats[enc_ind] = memory.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w).contiguous()
                if self.use_mask_head:
                    memory_for_mask_head =  memory.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w).contiguous()

                # print([x.is_contiguous() for x in proj_feats ])

        # broadcasting and fusion
        inner_outs = [proj_feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_high = inner_outs[0]
            feat_low = proj_feats[idx - 1]
            feat_high = self.lateral_convs[len(self.in_channels) - 1 - idx](feat_high)
            inner_outs[0] = feat_high
            upsample_feat = F.interpolate(feat_high, scale_factor=2., mode='nearest')
            if upsample_feat.shape != feat_low.shape:
                upsample_feat = F.pad(upsample_feat, [0, feat_low.shape[-1] - upsample_feat.shape[-1], 0, feat_low.shape[-2] - upsample_feat.shape[-2]])
            inner_out = self.fpn_blocks[len(self.in_channels)-1-idx](torch.concat([upsample_feat, feat_low], dim=1))
            inner_outs.insert(0, inner_out)

        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1): 
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]
            downsample_feat = self.downsample_convs[idx](feat_low)
            out = self.pan_blocks[idx](torch.concat([downsample_feat, feat_height], dim=1))
            if self.BMHA and cross_vp == False:
                h,w = out.shape[2:]
                query_mask_flatten = F.interpolate(query_mask_org[None].float(), size=(h,w)).to(torch.bool)[0]
                query_mask_flatten = query_mask_flatten.flatten(1)
                out_flatten = out.flatten(2).permute(0, 2, 1)
                _,support_feat = self.fusion_layers[index](out_flatten,support_feat,query_mask_flatten,support_mask)
                #out = out_flatten.permute(0, 2, 1).reshape(-1, self.hidden_dim, out.shape[2], out.shape[3]).contiguous()
            outs.append(out)
        prompt_dict = {"encoded_support":support_feat, 
                    "support_token_mask":support_mask}
        if self.use_mask_head:
            return outs,prompt_dict,memory_for_mask_head
        return outs,prompt_dict
    


class Tidev1Encoder(nn.Module):
    def __init__(
        self,
        num_layers,
        d_model=256,
        num_queries=300,
        enc_layer_share=False,
        nhead=8,
        dim_feedforward = 2048,
        dropout=0.01,
        num_feature_level=4,
        raw_support_feat_dim=384,
        fusion = True,
    ):
        """_summary_

        Args:
            encoder_layer (_type_): _description_
            num_layers (_type_): _description_
            norm (_type_, optional): _description_. Defaults to None.
            d_model (int, optional): _description_. Defaults to 256.
            num_queries (int, optional): _description_. Defaults to 300.
            enc_layer_share (bool, optional): _description_. Defaults to False.

        """
        super().__init__()
        # prepare layers
        self.layers = []
        self.fusion_layers = []
        self.support_input_proj = nn.Linear(raw_support_feat_dim, d_model,bias=True)
        encoder_layer = DeformableTransformerEncoderLayer(
            d_model,
            dim_feedforward,
            dropout,
            activation='relu',
            n_levels=num_feature_level
        )
        if fusion:
            feature_fusion_layer=BiAttentionBlock(
                v_dim=d_model,
                l_dim=d_model,
                embed_dim=dim_feedforward // 2,
                num_heads=nhead // 2,
                dropout=dropout,
                drop_path=0.0,
            )
        else:
            feature_fusion_layer = None
        if num_layers > 0:
            self.layers = _get_clones(
                encoder_layer, num_layers, layer_share=enc_layer_share
            )
            if feature_fusion_layer is not None:
                self.fusion_layers = _get_clones(
                    feature_fusion_layer, num_layers, layer_share=enc_layer_share
                )
            else:
                self.fusion_layers = None
        else:
            self.layers = []
            del encoder_layer

            if feature_fusion_layer is not None:
                self.fusion_layers = []
                del feature_fusion_layer

        self.query_scale = None
        self.num_queries = num_queries
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_feature_levels = num_feature_level
        self.level_embed = nn.Parameter(
                    torch.Tensor(num_feature_level, d_model)
                )

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device),
            )
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points
    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio
    def forward(
        self,
        # for images
        src: Tensor,
        pos: Tensor,
        key_padding_mask: Tensor,
        # for supports
        memory_support: Tensor = None,
        support_attention_mask: Tensor = None,
        pos_support: Tensor = None,
        position_ids: Tensor = None,
    ):
        """
        Input:
            - src: [bs, sum(hi*wi), 256]
            - pos: pos embed for src. [bs, sum(hi*wi), 256]
            - spatial_shapes: h,w of each level [num_level, 2]
            - level_start_index: [num_level] start point of level in sum(hi*wi).
            - valid_ratios: [bs, num_level, 2]
            - key_padding_mask: [bs, sum(hi*wi)]

            - memory_support: bs, n_support, 256
            - support_attention_mask: bs, n_support
                False for no padding; True for padding
            - pos_support: bs, n_support, 256

            - position_ids: bs, n_support
        Intermedia:
            - reference_points: [bs, sum(hi*wi), num_level, 2]
        Outpus:
            - output: [bs, sum(hi*wi), 256]
        """

        memory_support = self.support_input_proj(memory_support)
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(src, key_padding_mask, pos)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)

            src = src.flatten(2).transpose(1, 2)  # bs, hw, c
            mask = mask.flatten(1)  # bs, hw
            pos_embed = pos_embed.flatten(2).transpose(1, 2)  # bs, hw, c
            if self.num_feature_levels > 1 and self.level_embed is not None:
                lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            else:
                lvl_pos_embed = pos_embed
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)  # bs, \sum{hxw}, c
        mask_flatten = torch.cat(mask_flatten, 1)  # bs, \sum{hxw}
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)  # bs, \sum{hxw}, c
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=src_flatten.device
        )
        level_start_index = torch.cat(
            (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
        )
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in key_padding_mask], 1)
        output = src_flatten
        key_padding_mask = mask_flatten
        pos = lvl_pos_embed_flatten

        # preparation and reshape
        if self.num_layers > 0:
            reference_points = self.get_reference_points(
                spatial_shapes, valid_ratios, device=src.device
            )
        # main process
        for layer_id, layer in enumerate(self.layers):
            # if output.isnan().any() or memory_support.isnan().any():
            #     if os.environ.get('IPDB_SHILONG_DEBUG', None) == 'INFO':
            #         import ipdb; ipdb.set_trace()
            if self.fusion_layers:
                output, memory_support = self.fusion_layers[layer_id](
                    v=output,
                    l=memory_support,
                    attention_mask_v=key_padding_mask,
                    attention_mask_l=support_attention_mask,
                )
            # main process
            output = layer(
                src=output,
                pos=pos,
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                key_padding_mask=key_padding_mask,
            )
        prompt_dict = {"encoded_support":memory_support, 
                    "support_token_mask":support_attention_mask}
        return output,mask_flatten,prompt_dict,spatial_shapes,valid_ratios,lvl_pos_embed_flatten
class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model=256,
        d_ffn=1024,
        dropout=0.1,
        activation="relu",
        n_levels=4,
        n_heads=8,
        n_points=4,
    ):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(
            embed_dim=d_model,
            num_levels=n_levels,
            num_heads=n_heads,
            num_points=n_points,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation, d_model=d_ffn)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(
        self,
        src,
        pos,
        reference_points,
        spatial_shapes,
        level_start_index,
        key_padding_mask=None,
    ):
        # self attention
        # import ipdb; ipdb.set_trace()
        src2 = self.self_attn(
            query=self.with_pos_embed(src, pos),
            reference_points=reference_points,
            value=src,
            value_spatial_shapes=spatial_shapes,
            value_mask=key_padding_mask,
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        return src

