from dataclasses import dataclass,field
from typing import List
import torch.nn as nn
import torch


# from dataset.Dataset_Visual_Prompt import COCODatasetVisualPrompt
from dataset.Dataset_Visual_Prompt_resize import COCODatasetVisualPrompt

from torch.utils.data import dataset


import datetime
from typing import List
from misc import dist
import os
from timm.models import create_model




rank = dist.init_distributed()  
DATE = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")  


@dataclass
class args:
    version='v2'
    learnt_init_query = False
    val_cls_ids :list = field(default_factory=list)
    eval_only : bool = False
    cross_vp_flag : bool = False
    use_visual_prompt : bool = True
    shots : str = 'VP'
    
    test_save_path: str = '-result-project.csv'
    max_norm = 1
    fusion = True
    freeze_encoder = False
    num_feature_level = 4
    encoder_dim_feedforward: int = 1024
    decoder_dim_feedforward: int = 1024
    use_mixed_support_selection = False
    region_detect=False
    eos_coef : float = 1e-2#3e-2 #!!!CRUCIAL PARAMETER FOR PERFORMANCE!!!
    datasets:list = field(default_factory=list)
    num_queries=900
    gpu_num:int = 1
    model_name: str = 'obj365&openimage_swin_T_size1024_'

    output_dir: str = 'output_20250206_base_obj365_openimage'
    if os.path.exists(output_dir) is False:
        os.makedirs(output_dir,exist_ok=True)
    
    dataset_name: str = 'obj365,openimage'#,openimage
    # dataset_name: str = 'obj365'#mini debug
    
    
    test_ann_path: str = '/mnt/localdisk/tanm/dataset/CV_data/coco2017/annotations/instances_val2017.json'
    test_image_folder_path: str = '/mnt/localdisk/tanm/dataset/CV_data/coco2017/val2017'    
    
    # lvis_ann_path: str = '/mnt/localdisk/tanm/dataset/CV_data/lvis/lvis_v1_minival_inserted_image_name.json'
    lvis_ann_path: str = '/mnt/localdisk/tanm/dataset/CV_data/lvis/lvis_od_val.json'
    lvis_image_folder_path: str = '/mnt/localdisk/tanm/dataset/CV_data/coco2017'

    coco_ann_path: str = '/mnt/localdisk/tanm/dataset/CV_data/coco2017/annotations/instances_train2017.json'
    coco_image_folder_path: str = '/mnt/localdisk/tanm/dataset/CV_data/coco2017/train2017'

    obj365_ann_path: str = '/mnt/localdisk/tanm/dataset/CV_data/obj365/train/zhiyuan_objv2_train.json'
    # obj365_ann_path: str = '/mnt/localdisk/tanm/dataset/CV_data/obj365/train/100k_train.json'
    
    obj365_image_folder_path: str = '/mnt/localdisk/tanm/dataset/CV_data/obj365/train'

    openimage_ann_path: str ='/mnt/localdisk/tanm/dataset/CV_data/fiftyone/open-images-v7-coco/labels.json'
    openimage_image_folder_path: str = '/mnt/localdisk/tanm/dataset/CV_data/fiftyone/open-images-v7-coco/data'
    
    #==============================================================
    training: bool = True
    lr: float = 5e-5#0.000043
    lr_backbone: float =1e-5
    lr_drop: int = 6
    frozen_weights: List[str] = field(default_factory=list)
    lr_drop_gamma: float = 0.95
    weight_decay: float = 1e-4

    support_feat_dim: int = 256#512#384
    max_support_len: int = 100
    extra_shots: int = 20
    aug: bool = False
    strong_aug: bool = False

    batch_size: int = 10
    # batch_size: int = 16#swinv2_tiny_window16_256
    test_batch_size : int = 6
    num_workers: int = 7
    
    # resume:str = '/mnt/localdisk/tanm/codes/vp-osod/output_20250120/EPOCH-1-AP0_obj365&openimage_swin_T_size1024_-loss-2.038524391321923.pth'
    # resume:str = '/mnt/localdisk/tanm/codes/VisOpenDet/output_20250206_base/EPOCH_19_obj365_100k_swin_T_size1024__loss_1.5786968605995177.pth'
    resume:str = None
    backbone_freeze_stage : int = 5
    # backbone_freeze_stage : int = -1
    epochs: int = 20
    device: str = 'cuda'
    num_denoising: int = 0 #0 for no denoising
    BMHA: bool = False
    num_fusion_layers:int = 6
    #dataset_type: dataset.Dataset = COCODataset#COCODatasetAVG
    support_norm: bool = True
    query_support_norm: bool = False #True for CLIP-like similarity computation setting(cosine similarity as logits)
    #assert query_support_norm!=support_norm
    local_rank: int = 7 if rank else 0
    use_mask_head: bool = False
    losses: list = field(default_factory=list)
    weight_dict: dict = field(default_factory=dict)
    
    backbone: nn.Module = create_model('swinv2_tiny_window16_256', pretrained=True, 
                                       in_chans=3, features_only=True, 
                                       img_size=(1024, 1024),
                                       out_indices=(1, 2, 3))
    
    backbone_dims: List[int] = field(default_factory=list)
    use_align_loss:bool = False
    use_focal_loss:bool = False
    coco_test_dataset: dataset.Dataset = COCODatasetVisualPrompt#COCODatasetAVG
    coco_dataset: dataset.Dataset =COCODatasetVisualPrompt
    obj365_dataset: dataset.Dataset = COCODatasetVisualPrompt
    openimage_dataset: dataset.Dataset = COCODatasetVisualPrompt
    lvis_test_dataset:dataset.Dataset = COCODatasetVisualPrompt
args.val_cls_ids = None
args.backbone_dims = [192,384,768]#swinv2_tiny_window16_256

args.frozen_weights = None#['tide']
args.losses = [#'align',
               #'contrastive',
            #    'labels',
                'focal', 
                'boxes',
                # 'visual_text_align',
                # 'visual_text_align_rex',
                #"masks",
                ]
#['focal', 'masks']
args.weight_dict = {
                "loss_ce": 2, 
                "loss_bbox": 5, 
                "loss_giou": 2,   
                "loss_focal":2, 
                'cost_class': 2,
                'cost_bbox': 5,
                'cost_giou': 2,
                'loss_contrastive':1,
                'loss_align':1,
                'loss_mask':2,
                'loss_dice':2,
                'loss_visual_text_align':1,
                'loss_visual_text_align_rex':1,
                        }
if 'align' in args.losses:
    args.use_align_loss = True
if 'focal' in args.losses:
    args.use_focal_loss = True
 