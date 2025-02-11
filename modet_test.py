import os
model_download_path = 'https://hf-mirror.com'
os.environ['HF_ENDPOINT'] = model_download_path

import yaml
import torch
from backbone.uni_backbone import Backbone
from model.utils import NestedTensor


if __name__ == '__main__':
    with open('backbone.yaml', 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    model = Backbone(cfg)
    input = torch.randn(1, 3, 640, 640)
    mask = torch.randn(1, 640, 640)
    out = model(NestedTensor(input, mask))
    print(model)
    print(out)