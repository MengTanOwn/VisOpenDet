'''by tm
视觉指令进行开放目标检测;
消融实验，分析没有视觉特征选择的情景
'''

import copy
import torch 
from torch import Tensor
import torch.nn as nn 
import torch.nn.functional as F 
from .BMHA import BiAttentionBlock
from .decoder import MSDeformableAttention as MSDeformAttn 
from .prompt_encoder import PositionEmbeddingRandom,PositionEmbeddingSine
from .utils import get_activation,_get_clones,get_sine_pos_embed,_get_activation_fn
import random
import torchvision
from functools import reduce

def merge_dicts_reduce(list_of_dicts):
    return reduce(lambda a, b: {**a, **b}, list_of_dicts)

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
                 hidden_dim=512,
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
        self.max_len_support_box = max_support_len
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

            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
            self.cls_token_ref_point = nn.Parameter(torch.zeros(1,1,4))

            self.use_visual_prompt = use_visual_prompt
            
            # self.cross_attention_vp = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout)
            self.cross_attention_vp = MSDeformAttn(hidden_dim, nhead, 3, 4)
            self.cross_attention_vp_dropout = (
                nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            )
            self.cross_attention_vp_norm = nn.LayerNorm(hidden_dim)
            self.self_attn = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout,batch_first=True)
            self.dropout_post = nn.Dropout(dropout)
            self.norm_post = nn.LayerNorm(hidden_dim)
        
        self.out_channels = [hidden_dim for _ in range(len(in_channels))]
        self.out_strides = feat_strides

        # self.gap = nn.AdaptiveAvgPool2d((1, 1))  # 适应性全局平均池化
        # self.fc = nn.Linear(in_channels[0], hidden_dim)
        # self.gap_act = nn.GELU()
        
        # self.text_model_fintune = nn.Sequential(
        #     nn.Linear(1024, hidden_dim),
        #     nn.GELU(),
        #     nn.LayerNorm(hidden_dim,)
        # )
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
        # self.text_input_proj = nn.Linear(1024, hidden_dim)

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
    
    def _embed_boxes_add_feat(self, boxes: torch.Tensor,input_image_size,feat) -> torch.Tensor:
        """Embeds box prompts."""
        boxes_cl = boxes.clone()
        boxes = boxes + 0.5  # Shift to center of pixel
        coords = boxes.reshape(-1, 2, 2)
        corner_embedding = self.pe_layer.forward_with_coords(coords,input_image_size)
        corner_embedding[:, 0, :] += self.point_embeddings[2].weight
        corner_embedding[:, 1, :] += self.point_embeddings[3].weight
        #average the two corners
        corner_embedding = corner_embedding.mean(dim=1)

        feat = feat[None,:]
        downsampling_rate = input_image_size[0]//feat.shape[-1]
        for box_index,boxes_cl_i in enumerate(boxes_cl[0]):
            box_feature_map = [
                boxes_cl_i[0] / downsampling_rate,  # x_min
                boxes_cl_i[1] / downsampling_rate,  # y_min
                boxes_cl_i[2] / downsampling_rate,  # x_max
                boxes_cl_i[3] / downsampling_rate   # y_max
            ]
            
            # 将边界框坐标转换为 PyTorch 张量，并且添加一个 batch index（这里只有一个边界框）
            boxes = torch.tensor([0,*box_feature_map], dtype=torch.float32).view(1, -1).to(feat.device)  # 假设 batch size 为 1
            # boxes = boxes[None,:] # 假设 batch size 为 1
            
            # 特征图的 batch size
            batch_size = 1

            # 定义输出特征图的大小(h,w)
            # output_size = (max(int(box_feature_map[2]-box_feature_map[0]),1),max(int(box_feature_map[3]-box_feature_map[1]),1))  # 输出特征图的宽和高
            output_size = (max(int(box_feature_map[3]-box_feature_map[1]),1),max(int(box_feature_map[2]-box_feature_map[0]),1))  # 输出特征图的宽和高

            # 执行 RoI Align
            
            cropped_features = torchvision.ops.roi_align(feat, boxes, output_size, spatial_scale=1.0 )
            # print(cropped_features.shape)
            # print(torch.isnan(cropped_features).any())
            if torch.isnan(cropped_features).any():
                continue
            cropped_features = self.gap_act(self.gap(cropped_features))  # 应用全局平均池化
            cropped_features = cropped_features.view(cropped_features.size(0), -1)  # 展平特征图
            cropped_features = self.fc(cropped_features)  
            a = corner_embedding[box_index]   
            b = cropped_features[0]
            corner_embedding[box_index] =0.5*a+0.5*b
            # corner_embedding[box_index] =b
            # corner_embedding[box_index] += cropped_features[0]

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
    
    def _get_features(self, feats):
        feat_flatten = []
        spatial_shapes = []
        level_start_index = [0, ]
        for i, feat in enumerate(feats):
            _, _, h, w = feat.shape
            # [b, c, h, w] -> [b, h*w, c]
            feat_flatten.append(feat.flatten(2).permute(0, 2, 1))
            # [num_levels, 2]
            spatial_shapes.append([h, w])
            # [l], start index of each level
            level_start_index.append(h * w + level_start_index[-1])

        # [b, l, c]
        feat_flatten = torch.concat(feat_flatten, 1)
        level_start_index.pop()
        return (feat_flatten, spatial_shapes, level_start_index)
    
    def visual_prompt_cross_attention(self, support_feat,memory, query_mask_flatten,spatial_shapes=None,ref_points=None):
        Q = self.content_embedding.weight[None, :]
        #expand to the same size as support_feat
        Q = Q.expand(support_feat.shape[0], support_feat.shape[1], support_feat.shape[2])
        Q_ = self.cross_attention_vp(self.with_pos_embed(Q,support_feat), ref_points, memory,spatial_shapes,query_mask_flatten)[0]
        #Q_ = Q
        Q = Q + self.cross_attention_vp_dropout(Q_)
        Q = self.cross_attention_vp_norm(Q)
        q = k = self.with_pos_embed(Q, support_feat)
        Q_, _ = self.self_attn(q, k, value=Q, attn_mask=None)
        Q = Q + self.dropout_post(Q_)
        support_feat = self.norm_post(Q)
        return support_feat

    def forward(self, feats,support_feat=None,query_mask=None,support_mask=None,text_features=None,vp=None,cross_vp=False):
        assert len(feats) == len(self.in_channels)
        if support_feat!=None:
            support_feat = support_feat.type_as(feats[0])
            if cross_vp == False:
                support_feat = support_feat.transpose(1,2)
                # support_feat = self.support_input_proj(support_feat) #bs num_cls 256
            else:
                support_feat = support_feat.transpose(1,2)
                # support_feat = self.support_input_proj(support_feat)

            if not self.training and text_features is not None:
                text_features.tensors[:,:,0] = torch.zeros_like(text_features.tensors[:,:,0])
        elif vp!=None:
            # print('$$$$$$$$$$$$vp:',[each['boxes'] for each_index,each in enumerate(vp)])
            bs = feats[0].shape[0]
            support_feat = torch.zeros((bs,self.max_support_len, self.hidden_dim), device=feats[0].device)
            support_avg_feat = torch.zeros((bs,self.max_support_len, self.hidden_dim), device=feats[0].device) 
            support_mask = torch.zeros((bs, self.max_support_len), device=feats[0].device).type_as(query_mask)    
            support_cate_mask = torch.zeros((bs, self.max_support_len), device=feats[0].device)    
            
            
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        
        if len(proj_feats) == 1:
            self.use_encoder_idx = [0]
        # encoder
        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                h, w = proj_feats[enc_ind].shape[2:]
                query_mask = F.interpolate(query_mask[None].float(), size=(h,w)).to(torch.bool)[0]
                # flatten [B, C, H, W] to [B, HxW, C]
                src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)
                # query_mask_flatten = query_mask.flatten(1)
                if self.training or self.eval_spatial_size is None:
                    pos_embed = self.build_2d_sincos_position_embedding(
                        w, h, self.hidden_dim, self.pe_temperature).to(src_flatten.device)
                else:
                    pos_embed = getattr(self, f'pos_embed{enc_ind}', None).to(src_flatten.device)#1,hw, 256
                    
                memory = self.encoder[i](src_flatten, pos_embed=pos_embed)   
                proj_feats[enc_ind] = memory.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w).contiguous()
                if self.use_mask_head:
                    memory_for_mask_head =  memory.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w).contiguous()
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
            outs.append(out)
        
        memory,spatial_shapes,level_start_index = self._get_features(outs)
        if cross_vp == False:
            box_ref = [each['boxes'] for each in vp]
            box_embeddings = [self._embed_boxes(each['boxes'],each['input_image_size']) for each in vp]
            
            # box_embeddings = [self._embed_boxes_add_feat(each['boxes'],each['input_image_size'],feats[0][each_index]) for each_index,each in enumerate(vp)]
            # 将 box_ref 转换为张量
            # box_ref_tensor = [torch.tensor(boxes, dtype=torch.float32,device=memory[0].device) for boxes in box_ref]
            box_ref_tensor = [boxes.clone().detach().to(dtype=torch.float32, device=memory[0].device) for boxes in box_ref]
            # 获取图像的宽度和高度
            image_sizes = torch.tensor([each['input_image_size'] for each in vp], dtype=torch.float32,device=memory[0].device)
            normalized_box_ref = []
            for i, boxes in enumerate(box_ref_tensor):
                img_width, img_height = image_sizes[i]
                normalized_boxes = boxes / torch.tensor([img_width, img_height, img_width, img_height], dtype=torch.float32,device=memory[0].device)
                normalized_box_ref.append(normalized_boxes)

            max_len = self.max_len_support_box
            for i in range(len(box_embeddings)):
                box_embeddings[i] = F.pad(box_embeddings[i], (0, 0, 0, max_len - len(box_embeddings[i])))
                normalized_box_ref[i] = F.pad(normalized_box_ref[i], (0, 0, 0, max_len - len(normalized_box_ref[i][0])))
            box_embeddings = torch.stack(box_embeddings,dim=0)
            normalized_box_ref = torch.stack(normalized_box_ref,dim=0)
            normalized_box_ref = normalized_box_ref[:,0]
            corresponding_cates = [each['cates'] for each in vp]
            cls_token_ref_point = self.cls_token_ref_point.expand(bs, -1, -1)

            corresponding_cates_names = [each['cates2names'] for each in vp]
            bs_cate_feats = {}
            for i in range(bs):
                unique_cates = torch.unique(corresponding_cates[i])
                unique_cates_names = corresponding_cates_names[i]
                for cate in unique_cates:
                    cate_name = unique_cates_names[cate.item()]
                    mask = corresponding_cates[i] == cate
                    #pad mask
                    mask = F.pad(mask, (0, max_len - len(mask)))
                    #support_feat[i, cate] = box_embeddings[i, mask].mean(dim=0)
                    cur_all_class_embeddings = box_embeddings[i, mask].unsqueeze(0)
                    if self.use_visual_prompt and vp!=None:
                        cur_all_class_embeddings = self.support_input_proj(cur_all_class_embeddings)
                    temp_feat = torch.cat([self.cls_token.expand(cur_all_class_embeddings.shape[0], -1, -1), cur_all_class_embeddings], dim=1)
                    temp_feat = self.visual_prompt_cross_attention(temp_feat,memory[i].unsqueeze(0),query_mask_flatten=None,spatial_shapes=spatial_shapes,ref_points=torch.cat([cls_token_ref_point[i],normalized_box_ref[i,mask]]).unsqueeze(0).unsqueeze(2))

                    sp_cate_feat_mem = temp_feat[:,0][0]
                    support_feat[i, cate] = sp_cate_feat_mem
                    if cate_name not in bs_cate_feats.keys():
                        bs_cate_feats[cate_name] = [sp_cate_feat_mem]
                    else:
                        bs_cate_feats[cate_name].append(sp_cate_feat_mem)
                    support_cate_mask[i, cate] = 1
            
            #获取平均类别提示*start for train
            for i in range(bs):
                unique_cates_names = corresponding_cates_names[i]
                unique_cates = torch.unique(corresponding_cates[i])
                for cate in unique_cates:
                    cate_name = unique_cates_names[cate.item()]
                    avg_feat = torch.cat(bs_cate_feats[cate_name],dim=0).mean()
                    support_avg_feat[i,cate] = avg_feat
                    # if torch.rand(1) > 0.5:
                    #     support_feat[i, cate] = avg_feat

        if cross_vp == False:
            # support_mask = ~support_mask
            prompt_dict = {"encoded_support":support_feat, 
                        "support_token_mask":support_mask,
                        'support_avg_feat':support_avg_feat,
                        'support_cate_mask':support_cate_mask}
        else:
            # support_mask = ~support_mask
            prompt_dict = {"encoded_support":support_feat, 
                        "support_token_mask":support_mask
                        }
        if self.use_mask_head:
            return outs,prompt_dict,memory_for_mask_head
        return outs,prompt_dict
    
