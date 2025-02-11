import torch
from model.tide import TIDE
from model.encoder import HybridEncoder
from model.decoder import TideTransformer
from backbone.resnet import ResNet
from config import args
import collections
model = TIDE(   
            backbone=ResNet(18, pretrained=False,return_idx=[1,2,3]),
            encoder=HybridEncoder(in_channels=[128, 256, 512],
                                  BMHA=args.BMHA,
                                  dim_feedforward=1024,
                                  num_fusion_layers=args.num_fusion_layers,
                                  ),
            decoder=TideTransformer(num_classes=args.max_support_len,
                                    feat_channels=[256, 256, 256],
                                    num_denoising=args.num_denoising,
                                    ),
                                    multi_scale=None)
pretrained_dict = torch.load('output/EPOCH:16_BMHA.pth')#fusion_layers.0/BiAttn
#replace 'BiAttn' in pretrained_dict with fusion_layers.0
new_dict = collections.OrderedDict()
for each,value in pretrained_dict.items():
    if 'BiAttn' in each:
        each = each.replace('BiAttn','fusion_layers.0')
        print(each)
        new_dict[each] = value
    else:
        new_dict[each] = value
torch.save(new_dict,'pretrained.pth')

