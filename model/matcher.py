"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
Modules to compute the matching cost and solve the corresponding LSAP.

by lyuwenyu
"""
import numpy as np
import torch
import torch.nn.functional as F 

from scipy.optimize import linear_sum_assignment
from torch import nn

from .box_ops import box_cxcywh_to_xyxy, generalized_box_iou


def process_costs(C, sizes):
    # 将 C 转换为 numpy 数组以便使用 scipy 函数
    C_np = C
    
    # 检查矩阵中的无效数值
    if torch.isnan(C).any() or torch.isinf(C).any():
        print("Matrix contains invalid numeric entries.")
        # 将 NaN 替换为一个很大的值，例如正无穷
        C_np = np.where(np.isnan(C_np), np.inf, C_np)
        # 将 inf 替换为一个合理的最大值
        C_np = np.where(np.isinf(C_np), np.finfo(C_np.dtype).max, C_np)
    
    # 分割矩阵并计算指派
    indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C_np.split(sizes, -1))]
    
    return indices

def box_check(x):
    x_c, y_c, w, h = x.unbind(-1)
    # for i in range(x_c.shape[0])
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    box = torch.stack(b, dim=-1)
    non_zero_indices = torch.nonzero(x, as_tuple=False)[:, 0]
    non_empty_x = box[non_zero_indices]
    return non_empty_x

class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, weight_dict, use_focal_loss=False, alpha=0.25, gamma=2.0):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = weight_dict['cost_class']
        self.cost_bbox = weight_dict['cost_bbox']
        self.cost_giou = weight_dict['cost_giou']

        self.use_focal_loss = use_focal_loss
        self.alpha = alpha
        self.gamma = gamma

        assert self.cost_class != 0 or self.cost_bbox != 0 or self.cost_giou != 0, "all costs cant be 0"

    # @torch.no_grad()
    # def safe_generalized_box_iou(self,out_bbox, tgt_bbox):
    #     out_bbox_xyxy = box_cxcywh_to_xyxy(out_bbox)
    #     tgt_bbox_xyxy = box_cxcywh_to_xyxy(tgt_bbox)
        
    #     # 修正无效框
    #     out_bbox_xyxy[:, 2:] = torch.max(out_bbox_xyxy[:, 2:], out_bbox_xyxy[:, :2])
    #     tgt_bbox_xyxy[:, 2:] = torch.max(tgt_bbox_xyxy[:, 2:], tgt_bbox_xyxy[:, :2])
        
    #     # 计算 IOU
    #     giou = generalized_box_iou(out_bbox_xyxy, tgt_bbox_xyxy)
    #     return giou

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        if self.use_focal_loss:
            out_prob = F.sigmoid(outputs["pred_logits"].flatten(0, 1))
        else:
            out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]

        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]
        # out_bbox = box_check(out_bbox)
        # out_bbox = torch.nan_to_num(out_bbox, nan=0.01)


        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        if self.use_focal_loss:
            out_prob = out_prob[:, tgt_ids]
            neg_cost_class = (1 - self.alpha) * (out_prob**self.gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost_class = self.alpha * ((1 - out_prob)**self.gamma) * (-(out_prob + 1e-8).log())
            cost_class = pos_cost_class - neg_cost_class        
        else:
            cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes
        
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
        
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        # try:
        #     cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
        #     C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        # except:
        #     C = self.cost_bbox * cost_bbox + self.cost_class * cost_class
        # cost_giou = -self.safe_generalized_box_iou(out_bbox, tgt_bbox)
            # Final cost matrix
        # C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]

        # all_indices = [] 
        # start_idx = 0
        # for i, c in enumerate(C.split(sizes, -1)):
        #     # 提取子矩阵
        #     sub_matrix = c[i]
            
        #     # 确保处理的是三维数组
        #     if len(sub_matrix.shape) == 3:
        #         batch_size, num_rows, num_cols = sub_matrix.shape
                
        #         # 对每个批次单独应用 linear_sum_assignment
        #         batch_indices = []
        #         for batch in range(batch_size):
        #             # 获取当前批次的数据
        #             current_batch = sub_matrix[batch]
                    
        #             # 清理子矩阵中的无效数值
        #             cleaned_batch = torch.nan_to_num(current_batch, nan=0.0, posinf=None, neginf=None)
                    
        #             # 应用 linear_sum_assignment
        #             row_ind, col_ind = linear_sum_assignment(cleaned_batch.cpu().numpy())
                    
        #             # 添加结果到 batch_indices 列表
        #             batch_indices.append((row_ind, col_ind))
                
        #         # 添加批次结果到 all_indices
        #         all_indices.extend(batch_indices)
            
        #     else:
        #         # 如果不是三维，则直接处理
        #         cleaned_sub_matrix = torch.nan_to_num(sub_matrix, nan=0.0, posinf=None, neginf=None)
        #         row_ind, col_ind = linear_sum_assignment(cleaned_sub_matrix.cpu().numpy())
        #         all_indices.append((row_ind, col_ind))
            
           
            

        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        indices = process_costs(C, sizes)

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
