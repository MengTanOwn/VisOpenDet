'''
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
https://github.com/facebookresearch/detr/blob/main/util/box_ops.py
'''

import torch
from torchvision.ops.boxes import box_area

# import pprint
# pp = pprint.PrettyPrinter(indent=4)


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    # for i in range(x_c.shape[0])
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    # box = torch.stack(b, dim=-1)
    # non_zero_indices = torch.nonzero(x, as_tuple=False)[:, 0]
    # non_empty_x = box[non_zero_indices]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union

def process_boxes(boxes):
    # 检查边界框是否有效
    if not (boxes[:, 2:] >= boxes[:, :2]).all():
        # 获取无效的边界框索引
        invalid_indices = (boxes[:, 2:] < boxes[:, :2]).any(dim=1)
        
        # 打印或记录无效的边界框
        print(f"Invalid bounding boxes found at indices: {torch.where(invalid_indices)}")
        print(f"Invalid bounding boxes: {boxes[invalid_indices]}")

        # 选择处理策略
        # 选项 1: 丢弃无效的边界框
        boxes = boxes[~invalid_indices]
        
        # 选项 2: 修正无效的边界框
        # boxes[invalid_indices, 2:] = boxes[invalid_indices, :2]
        
        # 选项 3: 标记无效的边界框
        # boxes[invalid_indices, 2:] = -1  # 或其他标志值
        
    return boxes

def fix_boxes(boxes):
    # 确保框的右下角坐标大于等于左上角坐标
    # 使用克隆来避免原地操作
    boxes = boxes.clone()
    
    # 获取框的左上角和右下角坐标
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    
    # 修正右下角坐标，确保它们大于等于左上角坐标
    boxes[:, 2] = torch.max(x2, x1)
    boxes[:, 3] = torch.max(y2, y1)
    
    return boxes

def check_boxes_validity(boxes1):
    # 打印所有的框
    if not (boxes1[:, 2:] >= boxes1[:, :2]).all():
        print('##################################################')
        print(boxes1.shape)
        invalid_boxes_mask = (boxes1[:, 2:] < boxes1[:, :2]).any(dim=1)
        invalid_boxes = boxes1[invalid_boxes_mask]
        print("Invalid boxes found:")
        print(invalid_boxes)
        # print(f"All boxes:\n{boxes1}")
        


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
   
    # boxes1 = fix_boxes(boxes1)
    # if not (boxes1[:, 2:] >= boxes1[:, :2]).all():
    #     print('##################################################')
    #     print(boxes1)
    # check_boxes_validity(boxes1)
    # check_boxes_validity(boxes2)
    
    # boxes1 = torch.clamp(boxes1, min=0.0000001)
    # boxes1 = process_boxes(boxes1)
    # if not (boxes1[:, 2:] >= boxes1[:, :2]).all():
    #     print('##################################################')
    #     print(boxes1.shape)
    #     invalid_boxes_mask = (boxes1[:, 2:] < boxes1[:, :2]).any(dim=1)
    #     invalid_boxes = boxes1[invalid_boxes_mask]
    #     print("Invalid boxes found:")
    #     print(invalid_boxes)
    # boxes1 = torch.nan_to_num(boxes1, nan=0.5)
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all(),(boxes1[:1000],boxes2[:1000])
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)