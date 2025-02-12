from typing import List
from timm.models import create_model
import torch.nn as nn


class args:
    use_visual_prompt : bool = True
    max_norm = 1
    fusion = True
    freeze_encoder = False
    num_feature_level = 4
    encoder_dim_feedforward: int = 1024
    decoder_dim_feedforward: int = 1024
    use_mixed_support_selection = False
    region_detect=False
    eos_coef : float = 1e-2
    num_queries=900
    training: bool = False
    support_feat_dim: int = 256
    max_support_len: int = 100
    
    resume:str = 'model_weights/visopendet_base.pth'
    device: str = 'cpu'
    num_fusion_layers:int = 6
    support_norm: bool = True
    query_support_norm: bool = False 
    
    backbone: nn.Module = create_model('swinv2_tiny_window16_256', pretrained=True, 
                                       in_chans=3, features_only=True, 
                                       img_size=(1024, 1024),
                                       out_indices=(1, 2, 3))
    backbone_dims: List[int] = [192,384,768]