'''by tm
文本编码器的训练 结合ovido的neck
'''
import itertools
import copy
import numpy as np
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

from typing import Any
from transformers import AutoTokenizer, BertConfig, BertModel

def merge_dicts_reduce(list_of_dicts):
    return reduce(lambda a, b: {**a, **b}, list_of_dicts)

class ConvNormAct(nn.Module):
    """Utility module that stacks one convolution 2D layer,
    a normalization layer and an activation function.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        kernel_size (int): Size of the convolving kernel. Default: 1.
        stride (int): Stride of convolution. Default: 1.
        padding (int): Padding added to all four sides of the input. Default: 0.
        dilation (int): Spacing between kernel elements. Default: 1.
        groups (int): Number of blocked connections from input channels
            to output channels. Default: 1.
        bias (bool): if True, adds a learnable bias to the output. Default: True.
        norm_layer (nn.Module): Normalization layer used in `ConvNormAct`. Default: None.
        activation (nn.Module): Activation layer used in `ConvNormAct`. Default: None.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        norm_layer: nn.Module = None,
        activation: nn.Module = None,
        **kwargs,
    ):
        super(ConvNormAct, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            **kwargs,
        )
        self.norm = norm_layer
        self.activation = activation

    def forward(self, x):
        """Forward function for `ConvNormAct`"""
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class ChannelMapper(nn.Module):
    
    def __init__(
        self,
        in_channels = [512, 1024, 2048],
        out_channels = 256,
        kernel_size: int = 3,
        stride: int = 1,
        bias: bool = True,
        groups: int = 1,
        dilation: int = 1,
        norm_layer: nn.Module = None,
        activation: nn.Module = None,
    ):
        super(ChannelMapper, self).__init__()
        
        self.convs = nn.ModuleList()
        for in_channel in in_channels:
            self.convs.append(
                ConvNormAct(
                    in_channels=in_channel,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=(kernel_size - 1) // 2,
                    bias=bias,
                    groups=groups,
                    dilation=dilation,
                    norm_layer=copy.deepcopy(norm_layer),
                    activation=copy.deepcopy(activation),
                )
            )


    def forward(self, inputs):
        """Forward function for ChannelMapper

        Args:
            inputs (list[torch.Tensor]): The backbone feature maps.

        Return:
            tuple(torch.Tensor): A tuple of the processed features.
        """
        assert len(inputs) == len(self.convs)
        # outs = [self.convs[i](inputs[self.in_features[i]]) for i in range(len(inputs))]
        outs = [self.convs[i](inputs[i]) for i in range(len(inputs))]
        return outs

class BERTTokenizer:
    """BERT tokenizer.
    Args:
        tokenizer_name (str): the name of tokenizer.
        padding_mode (int): the padding mode of tokenizer.
            'max_length': padding the tokenined token to the max_length.
            'longest': padding the tokenized token to the length of the longest token.
        contex_length (int): the max context_length of tokenizer
            if given padding_mode is 'max_length'. Default is 'longest'.
        bos_token_id (int): the bos_token_id of tokenizer.
        eos_token_id (int): the eos_token_id of tokenizer.
        pad_token_id (int): the pad_token_id of tokenizer.
        dot_token_id (int): the dot_token_id of tokenizer.
    Example:
    """

    def __init__(
        self,
        tokenizer_name="bert-base-uncased",
        padding_mode="longest",
        context_length=48,
        dot_token_id=1012,
        bos_token_id=101,
        eos_token_id=102,
        pad_token_id=0,
    ) -> None:
        super().__init__()
        # BERT tokenizer in Huggingface, the bos_token_id and eos_token_id is None, we need to define them here.
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.padding_mode = padding_mode
        self.context_length = context_length
        self.dot_token_id = dot_token_id
        self.bos_token_id = (
            bos_token_id
            if self.tokenizer.bos_token_id is None
            else self.tokenizer.bos_token_id
        )
        self.eos_token_id = (
            eos_token_id
            if self.tokenizer.eos_token_id is None
            else self.tokenizer.eos_token_id
        )
        self.pad_token_id = (
            pad_token_id
            if self.tokenizer.pad_token_id is None
            else self.tokenizer.pad_token_id
        )

    def __call__(self, x, return_mask=False, *args: Any, **kwargs: Any) -> Any:
        tokenized_batch = self.tokenizer.batch_encode_plus(
            x,
            *args,
            max_length=self.context_length,
            padding=self.padding_mode,
            return_special_tokens_mask=True,
            return_tensors="pt",
            truncation=True,
            **kwargs,
        )
        output = {"input_ids": tokenized_batch["input_ids"]}
        if return_mask:
            output["attention_mask"] = tokenized_batch["attention_mask"]
        return output


class BERTEncoder(nn.Module):
    def __init__(
        self,
        tokenizer_cfg=dict(tokenizer_name="bert-base-uncased"),
        model_name="bert-base-uncased",
        output_dim=256,
        padding_mode="longest",
        context_length=48,
        pooling_mode="max",
        post_tokenize=False,
        is_normalize=True,
        is_proj=True,
        is_freeze=False,
        return_dict=False,
    ) -> None:
        super().__init__()
        assert pooling_mode in ["max", "mean", None]
        self.bos_token_id = 101
        self.eos_token_id = 102
        self.padding_mode = padding_mode
        self.context_length = context_length
        self.post_tokenize = post_tokenize
        self.tokenizer = BERTTokenizer(**tokenizer_cfg)
        lang_model_config = BertConfig.from_pretrained(model_name)
        self.lang_model = BertModel.from_pretrained(
            model_name, add_pooling_layer=False, config=lang_model_config
        )
        self.is_normalize = is_normalize
        self.pooling_mode = pooling_mode
        self.is_proj = is_proj
        self.is_freeze = is_freeze
        self.return_dict = return_dict
        self.num_layers = 1
        if self.is_proj:
            self.text_porj = nn.Parameter(
                torch.empty(self.lang_model.config.hidden_size, output_dim)
            )
            nn.init.normal_(
                self.text_porj, std=self.lang_model.config.hidden_size**-0.5
            )
            # self.text_porj = nn.Linear(self.lang_model.config.hidden_size, output_dim)
        # freeze parameters
        if self.is_freeze:
            print(f"Freezee parameters of {self.__class__.__name__}.")
            for param in self.lang_model.parameters():
                param.requires_grad = False

    def forward(self, x, *args, **kwargs):
        """Forward function of text_encoder.
        Args:
            x (list[str] or Tensor): the input text that is a list of category name(or definition) or toneized token.
                shape: N x [C, ] or [N, C, L]([N, L]), where L is the context_length.
        Returns:
            output (Tensor): the extracted text feature, shape: [N*C, D].
        """
        if self.post_tokenize:
            tokenized_batch = self.tokenizer(x, return_mask=True)
            input_ids = tokenized_batch["input_ids"].cuda()
            attention_mask = tokenized_batch["attention_mask"].cuda()
        else:
            assert self.context_length == x.shape[-1]
            input_ids = x.reshape(-1, self.context_length)
            attention_mask = (input_ids != self.tokenizer.pad_token_id).float()
        output = self.lang_model(
            input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, *args, **kwargs
        )["last_hidden_state"]
        output_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "last_hidden_state": output,
        }
        if self.pooling_mode == "mean":
            output = torch.mean(output, dim=1, keepdim=False)
            output_dict.update({"pooled_output": output})
        elif self.pooling_mode == "max":
            # take features from the eos_token embedding
            eot_indices = torch.nonzero(
                torch.eq(input_ids, self.tokenizer.eos_token_id)
            )
            output = output[torch.arange(output.shape[0]), eot_indices[:, 1]]
            output_dict.update({"pooled_output": output})
        else:
            raise NotImplementedError("Only support pooling_mode: [max, mean].")

        if self.is_normalize:
            output = F.normalize(output, p=2, dim=-1)
            output_dict.update({"normalized_output": output})
        if self.is_proj:
            output = output @ self.text_porj
            output_dict.update({"projected_output": output})
        if self.return_dict:
            return output_dict
        return output


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
        self.use_visual_prompt = use_visual_prompt
        self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = use_encoder_idx
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.eval_spatial_size = eval_spatial_size
        self.BMHA = BMHA
        self.use_mask_head = use_mask_head

        self.neck = ChannelMapper(in_channels = in_channels,
                                    out_channels = hidden_dim,
                                    kernel_size=1,
                                    norm_layer=(nn.GroupNorm)(num_groups=32, num_channels=hidden_dim),
                                    )
        # if use_visual_prompt:
        #     self.pe_layer = PositionEmbeddingRandom(hidden_dim // 2)
        #     self.num_point_embeddings: int = 4  # pos/neg point + 2 box corners
        #     point_embeddings = [nn.Embedding(1, hidden_dim) for i in range(self.num_point_embeddings)]
        #     self.point_embeddings = nn.ModuleList(point_embeddings)
        if self.use_visual_prompt:
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

            self.gap = nn.AdaptiveAvgPool2d((1, 1))  # 适应性全局平均池化
            self.fc = nn.Linear(in_channels[0], hidden_dim)
            self.gap_act = nn.GELU()
        if self.use_text:
            self.language_backbone = BERTEncoder(
                                            tokenizer_cfg=dict(tokenizer_name="bert-base-uncased"),
                                            model_name="bert-base-uncased",
                                            output_dim=hidden_dim,
                                            padding_mode="longest",
                                            context_length=48,
                                            pooling_mode="mean",
                                            post_tokenize=True,
                                            is_freeze=False,)
        self.out_channels = [hidden_dim for _ in range(len(in_channels))]
        self.out_strides = feat_strides

        # self._reset_parameters()

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
            support_cate_feat = torch.zeros((bs,self.max_support_len, self.hidden_dim), device=feats[0].device)
            
            support_mask = torch.zeros((bs, self.max_support_len), device=feats[0].device).type_as(query_mask)    
            support_cate_mask = torch.zeros((bs, self.max_support_len), device=feats[0].device)    
        cate_name_emb = {}    
        if self.use_text:
            catenames = [item['cates2names'].values() for item in vp] 
            catenames = list(itertools.chain(*catenames))
            # unique_catenames = list(set(catenames))
            unique_catenames, inverse = np.unique(catenames, return_inverse=True)
            # 创建掩码字典
            # mask_dict = {name: inverse == i for i, name in enumerate(unique_catenames)}
            unique_catenames_temple = ['a photo of '+i for i in unique_catenames]
            text_embed = self.language_backbone(unique_catenames_temple)
            # text_embed = [self.language_backbone([cate_temp])[0] for cate_temp in unique_catenames_temple]
            for index,catename in enumerate(unique_catenames):
                cate_name_emb[catename] = text_embed[index]
        
        outs = self.neck(feats)
        if cross_vp == False:           
            corresponding_cates = [each['cates'] for each in vp]    
            corresponding_cates_names = [each['cates2names'] for each in vp]
            for i in range(bs):
                unique_cates = torch.unique(corresponding_cates[i])
                # unique_cates_text_feat = corresponding_cates_feat[i]
                unique_cates_names = corresponding_cates_names[i]
                for cate in unique_cates:
                    cate_name = unique_cates_names[cate.item()]
                    cate_text_feat_i = cate_name_emb[cate_name]
                    support_feat[i, cate] = cate_text_feat_i
        if cross_vp == False:
            # support_mask = ~support_mask
            prompt_dict = {"encoded_support":support_feat, 
                        "support_token_mask":support_mask,
                        'support_cate_feat':support_cate_feat,
                        'support_cate_mask':support_cate_mask}
        else:
            # support_mask = ~support_mask
            prompt_dict = {"encoded_support":support_feat, 
                        "support_token_mask":support_mask
                        }
        return outs,prompt_dict
    
