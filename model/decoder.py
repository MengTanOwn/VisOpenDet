import math 
import copy 
from collections import OrderedDict

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.nn.init as init 

from .denoising import get_contrastive_denoising_training_group
from .utils import deformable_attention_core_func, get_activation, inverse_sigmoid,_get_activation_fn
from .utils import bias_init_with_prob,_get_clones,get_sine_pos_embed,gen_sineembed_for_position,gen_encoder_output_proposals
# from.ContrastiveEmbed import ContrastiveEmbed
from.ContrastiveEmbed import ClassEmbed
from torch import Tensor, nn
from typing import List, Optional




class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, act='relu'):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x



class MSDeformableAttention(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, num_levels=4, num_points=4,):
        """
        Multi-Scale Deformable Attention Module
        """
        super(MSDeformableAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.total_points = num_heads * num_levels * num_points

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.sampling_offsets = nn.Linear(embed_dim, self.total_points * 2,)
        self.attention_weights = nn.Linear(embed_dim, self.total_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

        self.ms_deformable_attn_core = deformable_attention_core_func

        self._reset_parameters()


    def _reset_parameters(self):
        # sampling_offsets
        init.constant_(self.sampling_offsets.weight, 0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True).values
        grid_init = grid_init.reshape(self.num_heads, 1, 1, 2).tile([1, self.num_levels, self.num_points, 1])
        scaling = torch.arange(1, self.num_points + 1, dtype=torch.float32).reshape(1, 1, -1, 1)
        grid_init *= scaling
        self.sampling_offsets.bias.data[...] = grid_init.flatten()

        # attention_weights
        init.constant_(self.attention_weights.weight, 0)
        init.constant_(self.attention_weights.bias, 0)

        # proj
        init.xavier_uniform_(self.value_proj.weight)
        init.constant_(self.value_proj.bias, 0)
        init.xavier_uniform_(self.output_proj.weight)
        init.constant_(self.output_proj.bias, 0)


    def forward(self,
                query,
                reference_points,
                value,
                value_spatial_shapes,
                value_mask=None):
        """
        Args:
            query (Tensor): [bs, query_length, C]
            reference_points (Tensor): [bs, query_length, n_levels, 2], range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area
            value (Tensor): [bs, value_length, C]
            value_spatial_shapes (List): [n_levels, 2], [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
            value_level_start_index (List): [n_levels], [0, H_0*W_0, H_0*W_0+H_1*W_1, ...]
            value_mask (Tensor): [bs, value_length], True for non-padding elements, False for padding elements

        Returns:
            output (Tensor): [bs, Length_{query}, C]
        """
        bs, Len_q = query.shape[:2]
        Len_v = value.shape[1]

        value = self.value_proj(value)
        if value_mask is not None:
            value_mask = (~value_mask).to(value.dtype).unsqueeze(-1)
            value *= value_mask
        value = value.reshape(bs, Len_v, self.num_heads, self.head_dim)

        sampling_offsets = self.sampling_offsets(query).reshape(
            bs, Len_q, self.num_heads, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query).reshape(
            bs, Len_q, self.num_heads, self.num_levels * self.num_points)
        attention_weights = F.softmax(attention_weights, dim=-1).reshape(
            bs, Len_q, self.num_heads, self.num_levels, self.num_points)

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.tensor(value_spatial_shapes)
            offset_normalizer = offset_normalizer.flip([1]).reshape(
                1, 1, 1, self.num_levels, 1, 2)
            sampling_locations = reference_points.reshape(
                bs, Len_q, 1, self.num_levels, 1, 2
            ) + sampling_offsets / offset_normalizer
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2] + sampling_offsets /
                self.num_points * reference_points[:, :, None, :, None, 2:] * 0.5)
        else:
            raise ValueError(
                "Last dim of reference_points must be 2 or 4, but get {} instead.".
                format(reference_points.shape[-1]))

        output = self.ms_deformable_attn_core(value, value_spatial_shapes, sampling_locations, attention_weights)

        output = self.output_proj(output)

        return output


class TransformerDecoderLayer(nn.Module):
    def __init__(self,
                 d_model=256,
                 n_head=8,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 n_levels=4,
                 n_points=4,):
        super(TransformerDecoderLayer, self).__init__()
        # support query cross attention 11-20 tiaoshi quxiao 
        self.ca_support = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.ca_support_dropout = (
            nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        )
        self.ca_support_norm = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # cross attention
        self.cross_attn = MSDeformableAttention(d_model, n_head, n_levels, n_points)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = getattr(F, activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        # self._reset_parameters()

    # def _reset_parameters(self):
    #     linear_init_(self.linear1)
    #     linear_init_(self.linear2)
    #     xavier_uniform_(self.linear1.weight)
    #     xavier_uniform_(self.linear2.weight)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        return self.linear2(self.dropout3(self.activation(self.linear1(tgt))))

    def forward(self,
                tgt,
                reference_points,
                memory,
                memory_spatial_shapes,
                memory_level_start_index,
                attn_mask=None,
                memory_mask=None,
                query_pos_embed=None,
                memory_support=None,
                memory_support_token_mask=None,
                ):
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos_embed)

        # if attn_mask is not None:
        #     attn_mask = torch.where(
        #         attn_mask.to(torch.bool),
        #         torch.zeros_like(attn_mask),
        #         torch.full_like(attn_mask, float('-inf'), dtype=tgt.dtype))

        tgt2, _ = self.self_attn(q, k, value=tgt, attn_mask=attn_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # support query cross attention11-20 tiaoshi quxiao 
        tgt2 = self.ca_support(
                    self.with_pos_embed(tgt.transpose(0,1), query_pos_embed.transpose(0,1)),  # query
                    memory_support.transpose(0,1),  # key
                    memory_support.transpose(0,1),  # value
                    key_padding_mask=memory_support_token_mask,#note that there's no ~
                )[0].transpose(0,1)

        tgt = tgt + self.ca_support_dropout(tgt2)
        tgt = self.ca_support_norm(tgt)

        # cross attention
        tgt2 = self.cross_attn(\
            self.with_pos_embed(tgt, query_pos_embed), 
            reference_points, 
            memory, 
            memory_spatial_shapes, 
            memory_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # ffn
        tgt2 = self.forward_ffn(tgt)
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)

        return tgt


class TransformerDecoder(nn.Module):
    def __init__(self, hidden_dim, decoder_layer, num_layers, eval_idx=-1):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx

    def forward(self,
                tgt,
                ref_points_unact,
                memory,
                memory_spatial_shapes,
                memory_level_start_index,
                bbox_head,
                score_head,
                query_pos_head,
                prompt_dict=None,
                attn_mask=None,
                memory_mask=None,
                use_mask_head=False,
                ):
        output = tgt
        dec_out_bboxes = []
        dec_out_logits = []
        # temp_scare = 0.07
        ref_points_detach = F.sigmoid(ref_points_unact)
        if use_mask_head:
            intermediate = []
            reference_points = ref_points_unact.sigmoid()
            ref_points = [reference_points] 
        for i, layer in enumerate(self.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)
            query_pos_embed = query_pos_head(ref_points_detach)

            output = layer(output, ref_points_input, memory,
                           memory_spatial_shapes, memory_level_start_index,
                           attn_mask, memory_mask, query_pos_embed,
                           prompt_dict['encoded_support'],
                           prompt_dict['support_token_mask'])
            
            inter_ref_bbox = F.sigmoid(bbox_head[i](output) + inverse_sigmoid(ref_points_detach))

            if self.training:
                score_head_ms,temp_scare = score_head[i](output,prompt_dict)
                dec_out_logits.append(score_head_ms)
                if i == 0:
                    dec_out_bboxes.append(inter_ref_bbox)
                else:
                    dec_out_bboxes.append(F.sigmoid(bbox_head[i](output) + inverse_sigmoid(ref_points)))

            elif i == self.eval_idx:
                score_head_ms,temp_scare = score_head[i](output,prompt_dict)
                dec_out_logits.append(score_head_ms)
                dec_out_bboxes.append(inter_ref_bbox)
                break

            ref_points = inter_ref_bbox
            ref_points_detach = inter_ref_bbox.detach(
            ) if self.training else inter_ref_bbox
        if use_mask_head:
            intermediate.append(output)
            return [
                    [itm_out.transpose(0, 1) for itm_out in intermediate],
                    [itm_refpoint.transpose(0, 1) for itm_refpoint in ref_points]
                ],torch.stack(dec_out_bboxes), torch.stack(dec_out_logits)

        return torch.stack(dec_out_bboxes), torch.stack(dec_out_logits),temp_scare



class TideTransformer(nn.Module):
    def __init__(self,
                 num_classes=81,
                 raw_support_feat_dim=384,
                 hidden_dim=512,
                 num_queries=300,
                 position_embed_type='sine',
                 feat_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 num_levels=3,
                 num_decoder_points=4,
                 nhead=8,
                 num_decoder_layers=6,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 num_denoising=100,
                 label_noise_ratio=0.5,
                 box_noise_scale=1.0,
                 learnt_init_query=False,
                 eval_spatial_size=None,
                 eval_idx=-1,
                 eps=1e-2, 
                 aux_loss=True,
                 normalize=False,
                 ):

        super(TideTransformer, self).__init__()
        self.use_mask_head = mask_head
        assert position_embed_type in ['sine', 'learned'], \
            f'ValueError: position_embed_type not supported {position_embed_type}!'
        assert len(feat_channels) <= num_levels
        assert len(feat_strides) == len(feat_channels)
        for _ in range(num_levels - len(feat_strides)):
            feat_strides.append(feat_strides[-1] * 2)

        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.feat_strides = feat_strides
        self.num_levels = num_levels
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.eps = eps
        self.num_decoder_layers = num_decoder_layers
        self.eval_spatial_size = eval_spatial_size
        self.aux_loss = aux_loss
        if align_loss and self.training:
            self.use_align_loss = True
            self.align_projection_image = nn.Linear(hidden_dim, hidden_dim//4)
            self.align_projection_support = nn.Linear(hidden_dim, hidden_dim//4)
        else:
            self.use_align_loss = False
        # backbone feature projection
        self._build_input_proj_layer(feat_channels,raw_support_feat_dim=raw_support_feat_dim)

        # Transformer module
        decoder_layer = TransformerDecoderLayer(hidden_dim, nhead, dim_feedforward, dropout, activation, num_levels, num_decoder_points)
        self.decoder = TransformerDecoder(hidden_dim, decoder_layer, num_decoder_layers, eval_idx)
        

        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale
        # denoising part
        if num_denoising > 0: 
            # self.denoising_class_embed = nn.Embedding(num_classes, hidden_dim, padding_idx=num_classes-1) # TODO for load paddle weights
            self.denoising_class_embed = nn.Embedding(num_classes+1, hidden_dim, padding_idx=num_classes)

        # decoder embedding
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_pos_head = MLP(4, 2 * hidden_dim, hidden_dim, num_layers=2)

        # encoder head
        self.enc_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim,)
        )
        self.enc_score_head = ClassEmbed(lang_embed_dim=hidden_dim,embed_dim=hidden_dim)
        
        #self.enc_score_head = nn.Linear(hidden_dim, num_classes)
        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 4, num_layers=3)

        # decoder head
        self.normal = normalize
        self.dec_score_head = nn.ModuleList([
                ClassEmbed(lang_embed_dim=hidden_dim,embed_dim=hidden_dim,return_logit_scare=True)
                for _ in range(num_decoder_layers)
            ])

        self.dec_bbox_head = nn.ModuleList([
            MLP(hidden_dim, hidden_dim, 4, num_layers=3)
            for _ in range(num_decoder_layers)
        ])

        #self.ln_post = nn.LayerNorm(hidden_dim)

        # init encoder output anchors and valid_mask
        if self.eval_spatial_size:
            self.anchors, self.valid_mask = self._generate_anchors()

        self._reset_parameters()

    def _reset_parameters(self):
        bias = bias_init_with_prob(0.01)

        #init.constant_(self.enc_score_head.bias, bias)
        init.constant_(self.enc_bbox_head.layers[-1].weight, 0)
        init.constant_(self.enc_bbox_head.layers[-1].bias, 0)

        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            #init.constant_(cls_.bias, bias)
            init.constant_(reg_.layers[-1].weight, 0)
            init.constant_(reg_.layers[-1].bias, 0)
        
        # linear_init_(self.enc_output[0])
        init.xavier_uniform_(self.enc_output[0].weight)
        if self.learnt_init_query:
            init.xavier_uniform_(self.tgt_embed.weight)
        init.xavier_uniform_(self.query_pos_head.layers[0].weight)
        init.xavier_uniform_(self.query_pos_head.layers[1].weight)


    def _build_input_proj_layer(self, feat_channels,raw_support_feat_dim=384):
        self.input_proj = nn.ModuleList()
        for in_channels in feat_channels:
            self.input_proj.append(
                nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(in_channels, self.hidden_dim, 1, bias=False)), 
                    ('norm', nn.BatchNorm2d(self.hidden_dim,))])
                )
            )

        in_channels = feat_channels[-1]

        for _ in range(self.num_levels - len(feat_channels)):
            self.input_proj.append(
                nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(in_channels, self.hidden_dim, 3, 2, padding=1, bias=False)),
                    ('norm', nn.BatchNorm2d(self.hidden_dim))])
                )
            )
            in_channels = self.hidden_dim
    def _get_encoder_input(self, feats):
        # get projection features
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        if self.num_levels > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_levels):
                if i == len_srcs:
                    proj_feats.append(self.input_proj[i](feats[-1]))
                else:
                    proj_feats.append(self.input_proj[i](proj_feats[-1]))

        # get encoder inputs
        feat_flatten = []
        spatial_shapes = []
        level_start_index = [0, ]
        for i, feat in enumerate(proj_feats):
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

    def _generate_anchors(self,
                          spatial_shapes=None,
                          grid_size=0.05,
                          dtype=torch.float32,
                          device='cpu'):
        if spatial_shapes is None:
            spatial_shapes = [[int(self.eval_spatial_size[0] / s), int(self.eval_spatial_size[1] / s)]
                for s in self.feat_strides
            ]
        anchors = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            grid_y, grid_x = torch.meshgrid(\
                torch.arange(end=h, dtype=dtype), \
                torch.arange(end=w, dtype=dtype), indexing='ij')
            grid_xy = torch.stack([grid_x, grid_y], -1)
            valid_WH = torch.tensor([w, h]).to(dtype)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH
            wh = torch.ones_like(grid_xy) * grid_size * (2.0 ** lvl)
            anchors.append(torch.concat([grid_xy, wh], -1).reshape(-1, h * w, 4))

        anchors = torch.concat(anchors, 1).to(device)
        valid_mask = ((anchors > self.eps) * (anchors < 1 - self.eps)).all(-1, keepdim=True)
        anchors = torch.log(anchors / (1 - anchors))
        # anchors = torch.where(valid_mask, anchors, float('inf'))
        # anchors[valid_mask] = torch.inf # valid_mask [1, 8400, 1]
        anchors = torch.where(valid_mask, anchors, torch.inf)

        return anchors, valid_mask


    def _get_decoder_input(self,
                           memory,
                           spatial_shapes,
                           prompt_dict=None,
                           denoising_class=None,
                           denoising_bbox_unact=None):
        bs, _, _ = memory.shape
        # prepare input for decoder
        if self.training or self.eval_spatial_size is None:
            anchors, valid_mask = self._generate_anchors(spatial_shapes, device=memory.device)
        else:
            anchors, valid_mask = self.anchors.to(memory.device), self.valid_mask.to(memory.device)

        # memory = torch.where(valid_mask, memory, 0)
        memory = valid_mask.to(memory.dtype) * memory  # TODO fix type error for onnx export 

        output_memory = self.enc_output(memory)

        enc_outputs_class = self.enc_score_head(output_memory,prompt_dict)
        enc_outputs_coord_unact = self.enc_bbox_head(output_memory) + anchors

        _, topk_ind = torch.topk(enc_outputs_class.max(-1).values, self.num_queries, dim=1)
        
        reference_points_unact = enc_outputs_coord_unact.gather(dim=1, \
            index=topk_ind.unsqueeze(-1).repeat(1, 1, enc_outputs_coord_unact.shape[-1]))

        enc_topk_bboxes = F.sigmoid(reference_points_unact)
        if denoising_bbox_unact is not None:
            reference_points_unact = torch.concat(
                [denoising_bbox_unact, reference_points_unact], 1)
        
        enc_topk_logits = enc_outputs_class.gather(dim=1, \
            index=topk_ind.unsqueeze(-1).repeat(1, 1, enc_outputs_class.shape[-1]))

        # extract region features
        if self.learnt_init_query:
            target = self.tgt_embed.weight.unsqueeze(0).tile([bs, 1, 1])
        else:
            target = output_memory.gather(dim=1, \
                index=topk_ind.unsqueeze(-1).repeat(1, 1, output_memory.shape[-1]))
            target = target.detach()

        if denoising_class is not None:
            target = torch.concat([denoising_class, target], 1)

        return target, reference_points_unact.detach(), enc_topk_bboxes, enc_topk_logits


    def forward(self, feats,prompt_dict,targets=None):

        # input projection and embedding
        (memory, spatial_shapes, level_start_index) = self._get_encoder_input(feats)
        #bs = memory.shape[0]
        #support_feat = self.support_input_proj(support_feat)
        #interpolate memory to the size of support_feat to fuse query and support features
        # target_size = support_feat.shape[1]
        # support_feat+=F.interpolate(memory.transpose(1,2), size=target_size).transpose(1,2)
        # prepare denoising training
        if self.training and self.num_denoising > 0:
            denoising_class, denoising_bbox_unact, attn_mask, dn_meta = \
                get_contrastive_denoising_training_group(targets, \
                    self.num_classes, 
                    self.num_queries, 
                    self.denoising_class_embed, 
                    num_denoising=self.num_denoising, 
                    label_noise_ratio=self.label_noise_ratio, 
                    box_noise_scale=self.box_noise_scale, )
        else:
            denoising_class, denoising_bbox_unact, attn_mask, dn_meta = None, None, None, None

        target, init_ref_points_unact, enc_topk_bboxes, enc_topk_logits = \
            self._get_decoder_input(memory, spatial_shapes, prompt_dict,denoising_class, denoising_bbox_unact)

        # decoder
        if self.use_mask_head:
            [hs,memory_decoded],out_bboxes, out_logits = self.decoder(
                target,
                init_ref_points_unact,
                memory,
                spatial_shapes,
                level_start_index,
                self.dec_bbox_head,
                self.dec_score_head,
                self.query_pos_head,
                prompt_dict=prompt_dict,
                attn_mask=attn_mask,
                use_mask_head=self.use_mask_head,
                )
            return hs,memory_decoded,out_bboxes, out_logits
        else:
            out_bboxes, out_logits, temp_scare= self.decoder(
                target,
                init_ref_points_unact,
                memory,
                spatial_shapes,
                level_start_index,
                self.dec_bbox_head,
                self.dec_score_head,
                self.query_pos_head,
                prompt_dict=prompt_dict,
                attn_mask=attn_mask,
                )

        if self.training and dn_meta is not None:
            dn_out_bboxes, out_bboxes = torch.split(out_bboxes, dn_meta['dn_num_split'], dim=2)
            dn_out_logits, out_logits = torch.split(out_logits, dn_meta['dn_num_split'], dim=2)
        if self.use_align_loss and self.training:
            query_image_features = self.align_projection_image(target)
            support_image_features = self.align_projection_support(prompt_dict['encoded_support'])
            out = {'pred_logits': out_logits[-1], 'pred_boxes': out_bboxes[-1],
                'query_image_features':query_image_features,'support_image_features':support_image_features}
        else:
            query_image_features = target
            support_image_features = prompt_dict['encoded_support']
            try:
                out = {'pred_logits': out_logits[-1], 
                    'pred_boxes': out_bboxes[-1],
                    'query_image_features':query_image_features,
                    'support_image_features':support_image_features,
                    'temp_scare':temp_scare,
                    'support_avg_feat':prompt_dict['support_avg_feat'],
                    'support_text_labels':prompt_dict['support_cate_mask']}
            except:
                out = {'pred_logits': out_logits[-1], 
                    'pred_boxes': out_bboxes[-1],
                    'query_image_features':query_image_features,
                    'support_image_features':support_image_features
                   }

        if self.training and self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(out_logits[:-1], out_bboxes[:-1],[query_image_features],[support_image_features])
            out['aux_outputs'].extend(self._set_aux_loss([enc_topk_logits], [enc_topk_bboxes],[query_image_features],[support_image_features]))
            # out['aux_outputs']['support_image_features']=support_image_features
            # out['aux_outputs']['support_text_features']=prompt_dict['support_cate_feat']
            # out['aux_outputs']['support_text_labels']=prompt_dict['support_cate_mask']
            if self.training and dn_meta is not None:
                out['dn_aux_outputs'] = self._set_aux_loss(dn_out_logits, dn_out_bboxes,[query_image_features],[support_image_features])
                out['dn_meta'] = dn_meta

        return out


    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord,q_feat=[None],s_feat=[None]):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b,'query_image_features':c,'support_image_features':d}
                for a, b,c,d in zip(outputs_class, outputs_coord,q_feat,s_feat)]



class Tidev1Decoder(nn.Module):
    def __init__(
        self,
        num_layers,
        num_classes=81,
        d_model=256,
        query_dim=4,
        num_feature_levels=1,
        num_queries=300,
        dim_feedforward = 2048,
    ):
        super().__init__()
        decoder_layer = DeformableTransformerDecoderLayer(d_model,d_ffn=dim_feedforward,n_levels=num_feature_levels)
        self.enc_output = nn.Linear(d_model, d_model)
        self.enc_output_norm = nn.LayerNorm(d_model)
        self.num_queries = num_queries
        self.tgt_embed = nn.Embedding(self.num_queries, d_model)
        nn.init.normal_(self.tgt_embed.weight.data)
        if num_layers > 0:
            self.layers = _get_clones(decoder_layer, num_layers)
        else:
            self.layers = []
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(d_model)
        self.query_dim = query_dim
        assert query_dim in [2, 4], "query_dim should be 2/4 but {}".format(query_dim)
        self.num_feature_levels = num_feature_levels

        self.ref_point_head = MLP(query_dim // 2 * d_model, d_model, d_model, 2)
        self.query_pos_sine_scale = None
        _class_embed = ContrastiveEmbed(max_support_len=num_classes)
        _bbox_embed = MLP(d_model, d_model, 4, 3)
        nn.init.constant_(_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_bbox_embed.layers[-1].bias.data, 0)

        box_embed_layerlist = [
            _bbox_embed for i in range(num_layers)
        ]
        class_embed_layerlist = [
            _class_embed for i in range(num_layers)
        ]
        self.bbox_embed = nn.ModuleList(box_embed_layerlist)
        self.class_embed = nn.ModuleList(class_embed_layerlist)
        self.query_scale = None
        self.enc_out_class_embed = copy.deepcopy(_class_embed)
        self.enc_out_bbox_embed = copy.deepcopy(_bbox_embed)
        self.d_model = d_model
        self.ref_anchor_head = None

    def forward(
        self,
        memory,
        mask_flatten,
        support_dict,
        spatial_shapes,
        level_start_index,
        valid_ratios,
        lvl_pos_embed_flatten,
        tgt_key_padding_mask=None,
    ):
        """
        Input:
            - tgt: nq, bs, d_model
            - memory: hw, bs, d_model
            - pos: hw, bs, d_model
            - refpoints_unsigmoid: nq, bs, 2/4
            - valid_ratios/spatial_shapes: bs, nlevel, 2
        """
        memory_support = support_dict["encoded_support"]
        bs = memory.shape[0]
        support_attention_mask = support_dict["support_token_mask"]
        refpoint_embed_,_,_,_,_ = self.support_guide_query_selection(
                memory, mask_flatten, spatial_shapes, support_dict
        )
        # gather tgt
        tgt_ = self.tgt_embed.weight[:, None, :].repeat(1, bs, 1).transpose(0, 1) # nq, bs, d_model
        refpoints_unsigmoid,output=refpoint_embed_.transpose(0,1),tgt_.transpose(0, 1)
        intermediate = []
        reference_points = refpoints_unsigmoid.sigmoid()
        ref_points = [reference_points]

        for layer_id, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = (
                    reference_points[:, :, None]
                    * torch.cat([valid_ratios, valid_ratios], -1)[None, :]
                )  # nq, bs, nlevel, 4
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = (
                    reference_points[:, :, None] * valid_ratios[None, :]
                )
            query_sine_embed = gen_sineembed_for_position(
                reference_points_input[:, :, 0, :]
            )  # nq, bs, 256*2
            #support_pos = self.support_embed.repeat(output.shape[1],1,1)[:,:memory_support.shape[1],:]
            bs, n_support, support_dim = memory_support.shape
            support_pos = (
                torch.arange(n_support, device=memory_support.device)
                .float()
                .unsqueeze(0)
                .unsqueeze(-1)
                .repeat(bs, 1, 1)
            )
            support_pos = get_sine_pos_embed(
                support_pos, num_pos_feats=self.d_model, exchange_xy=False
            )

            # conditional query
            raw_query_pos = self.ref_point_head(query_sine_embed)  # nq, bs, 256
            pos_scale = self.query_scale(output) if self.query_scale is not None else 1
            query_pos = pos_scale * raw_query_pos
            # if os.environ.get("SHILONG_AMP_INFNAN_DEBUG") == '1':
            #     if query_pos.isnan().any() | query_pos.isinf().any():
            #         import ipdb; ipdb.set_trace()

            # main process
            output = layer(
                tgt=output,
                tgt_pos=query_pos,
                tgt_query_sine_embed=query_sine_embed,
                tgt_key_padding_mask=tgt_key_padding_mask,
                tgt_reference_points=reference_points_input,
                memory_support=memory_support,
                support_attention_mask=support_attention_mask,
                support_pos = support_pos,
                memory=memory.transpose(0, 1),
                memory_key_padding_mask=mask_flatten,
                memory_level_start_index=level_start_index,
                memory_spatial_shapes=spatial_shapes,
                memory_pos=lvl_pos_embed_flatten.transpose(0, 1),
                self_attn_mask=None,
                cross_attn_mask=None,
            )
            if output.isnan().any() | output.isinf().any():
                print(f"output layer_id {layer_id} is nan")
                try:
                    num_nan = output.isnan().sum().item()
                    num_inf = output.isinf().sum().item()
                    print(f"num_nan {num_nan}, num_inf {num_inf}")
                except Exception as e:
                    print(e)
                    # if os.environ.get("SHILONG_AMP_INFNAN_DEBUG") == '1':
                    #     import ipdb; ipdb.set_trace()

            # iter update
            if self.bbox_embed is not None:
                # box_holder = self.bbox_embed(output)
                # box_holder[..., :self.query_dim] += inverse_sigmoid(reference_points)
                # new_reference_points = box_holder[..., :self.query_dim].sigmoid()

                reference_before_sigmoid = inverse_sigmoid(reference_points)
                delta_unsig = self.bbox_embed[layer_id](output)
                outputs_unsig = delta_unsig + reference_before_sigmoid
                new_reference_points = outputs_unsig.sigmoid()

                reference_points = new_reference_points.detach()
                # if layer_id != self.num_layers - 1:
                ref_points.append(new_reference_points)

            intermediate.append(self.norm(output))
        hs, references = [itm_out.transpose(0, 1) for itm_out in intermediate],[itm_refpoint.transpose(0, 1) for itm_refpoint in ref_points]
        # deformable-detr-like anchor update
        outputs_coord_list = []
        for dec_lid, (layer_ref_sig, layer_bbox_embed, layer_hs) in enumerate(
            zip(references[:-1], self.bbox_embed, hs)
        ):
            layer_delta_unsig = layer_bbox_embed(layer_hs)
            layer_outputs_unsig = layer_delta_unsig + inverse_sigmoid(layer_ref_sig)
            layer_outputs_unsig = layer_outputs_unsig.sigmoid()
            outputs_coord_list.append(layer_outputs_unsig)
        outputs_coord_list = torch.stack(outputs_coord_list)
        outputs_class = torch.stack(
                [
                    layer_cls_embed(layer_hs, support_dict)
                    for layer_cls_embed, layer_hs in zip(self.class_embed, hs)
                    # layer_cls_embed(F.normalize(layer_hs,dim=-1), prompt_dict) for layer_cls_embed, layer_hs in zip(self.class_embed, hs)
                ]
            )
        out = {
            "pred_logits": outputs_class[-1],
            "pred_boxes": outputs_coord_list[-1],
        }
        if self.training:
            out['aux_outputs'] = self._set_aux_loss(outputs_class[:-1], outputs_coord_list[:-1])
        return out
    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord,q_feat=[None],s_feat=[None]):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b,'query_image_features':c,'support_image_features':d}
                for a, b,c,d in zip(outputs_class, outputs_coord,q_feat,s_feat)]
    def support_guide_query_selection(
        self, memory, mask_flatten, spatial_shapes, support_dict
    ):
        output_memory, output_proposals = gen_encoder_output_proposals(
            memory, mask_flatten, spatial_shapes
        )
        bs = memory.shape[0]
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        enc_outputs_class_unselected = self.enc_out_class_embed(
            output_memory, support_dict  #(4,1200,256)@(4,13,256).t()->(4,1200,13)
        )
        topk_logits = enc_outputs_class_unselected.max(-1)[0]
        enc_outputs_coord_unselected = (
            self.enc_out_bbox_embed(output_memory) + output_proposals
        )#detr (4,1200,4)  nn.Linear(256,4)
        if topk_logits.shape[1] < self.num_queries:
            # 填补logits and coord防止topk报错
            topk_logits = torch.cat(
                [
                    topk_logits,
                    torch.zeros(bs, self.num_queries - topk_logits.shape[1]).to(
                        topk_logits.device
                    )
                    * 0,
                ],
                dim=1,
            )
            enc_outputs_coord_unselected = torch.cat(
                [
                    enc_outputs_coord_unselected,
                    torch.zeros(
                        bs, self.num_queries - enc_outputs_coord_unselected.shape[1], 4
                    ).to(enc_outputs_coord_unselected.device)
                    * 0,
                ],
                dim=1,
            )
        topk = self.num_queries
        topk_proposals = torch.topk(topk_logits, topk, dim=1)[1]  # bs, nq

        # gather boxes
        refpoint_embed_undetach = torch.gather(
            enc_outputs_coord_unselected,
            1,
            topk_proposals.unsqueeze(-1).repeat(1, 1, 4),
        )  # unsigmoid
        refpoint_embed_ = refpoint_embed_undetach.detach()
        init_box_proposal = None#torch.gather(output_proposals, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4)).sigmoid() # sigmoid
        return refpoint_embed_,init_box_proposal,output_memory,topk_proposals,refpoint_embed_undetach
class DeformableTransformerDecoderLayer(nn.Module):
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

        # cross attention
        self.cross_attn = MSDeformableAttention(
            embed_dim=d_model,
            num_levels=n_levels,
            num_heads=n_heads,
            num_points=n_points,
        )
        self.dropout1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm1 = nn.LayerNorm(d_model)

        # cross attention support
        self.ca_support = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.ca_support_dropout = (
            nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        )
        self.ca_support_norm = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation, d_model=d_ffn, batch_dim=1)
        self.dropout3 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm3 = nn.LayerNorm(d_model)

        self.key_aware_proj = None


    def rm_self_attn_modules(self):
        self.self_attn = None
        self.dropout2 = None
        self.norm2 = None

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos
    @staticmethod
    def with_pce_embed(tensor, pce, pce_indices, cur_support_len):
        # for i in range(len(pce_indices)):
        #     if pce_indices[i].shape[0]<cur_support_len:
        #         pce_indices[i] = torch.concat((pce_indices[i],torch.zeros(cur_support_len-pce_indices[i].shape[0],dtype=int).cuda()))
        for i in range(tensor.shape[0]):
            tensor[i] += pce(pce_indices[i])
        return tensor  # + torch.cat((pce(pce_indices),torch.zeros(tensor.shape[0],cur_support_len-pce_indices.shape[1],tensor.shape[-1]).cuda()),dim=1)

    def forward_ffn(self, tgt):
        with torch.cuda.amp.autocast(enabled=False):
            tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(
        self,
        # for tgt
        tgt: Optional[Tensor],  # nq, bs, d_model   (4,32,512)->encoder(4,32,512) 从特征到得分矩阵
        tgt_pos: Optional[Tensor] = None,  # pos for query. MLP(Sine(pos))   (4,32,512)->encoder(4,32,512) 正余弦编码
        tgt_query_sine_embed: Optional[Tensor] = None,  # pos for query. Sine(pos)
        tgt_key_padding_mask: Optional[Tensor] = None,
        tgt_reference_points: Optional[Tensor] = None,  # nq, bs, 4
        memory_support: Optional[Tensor] = None,  # bs, num_token, d_model
        support_attention_mask: Optional[Tensor] = None,  # bs, num_token
        support_pos:Optional[Tensor] = None,
        # for memory
        memory: Optional[Tensor] = None,  # hw, bs, d_model
        memory_key_padding_mask: Optional[Tensor] = None,
        memory_level_start_index: Optional[Tensor] = None,  # num_levels
        memory_spatial_shapes: Optional[Tensor] = None,  # bs, num_levels, 2
        memory_pos: Optional[Tensor] = None,  # pos for memory
        # sa
        self_attn_mask: Optional[Tensor] = None,  # mask used for self-attention
        cross_attn_mask: Optional[Tensor] = None,  # mask used for cross-attention
        # pseudo emebedding
        pce: Optional[Tensor] = None,  # TODO
        pce_indices: Optional[Tensor] = None,
        tgt_prompt:Optional[Tensor]=None,
    ):
        """
        Input:
            - tgt/tgt_query_pos: nq, bs, d_model
            -
        """
        assert cross_attn_mask is None

        # self attention
        if self.self_attn is not None:
            q = k = self.with_pos_embed(tgt, tgt_pos)
            #support self_attention
            tgt2 = self.self_attn(q, k, tgt, attn_mask=self_attn_mask)[0]
            tgt = tgt + self.dropout2(tgt2)
            tgt = self.norm2(tgt)


        tgt2 = self.ca_support(
            self.with_pos_embed(tgt, tgt_pos),  # query
            memory_support.transpose(0, 1),  # key
            memory_support.transpose(0, 1),  # value
            key_padding_mask=support_attention_mask,
        )[0]
        tgt = tgt + self.ca_support_dropout(tgt2)
        tgt = self.ca_support_norm(tgt)

        tgt2 = self.cross_attn(
            query=self.with_pos_embed(tgt, tgt_pos).transpose(0, 1),
            reference_points=tgt_reference_points.transpose(0, 1).contiguous(),
            value=memory.transpose(0, 1),
            value_spatial_shapes=memory_spatial_shapes,
            value_mask=memory_key_padding_mask,
        ).transpose(0, 1)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)
        return tgt
