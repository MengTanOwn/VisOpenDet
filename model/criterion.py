"""
reference: 
https://github.com/facebookresearch/detr/blob/main/models/detr.py

by lyuwenyu
"""


import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torchvision

# from torchvision.ops import box_convert, generalized_box_iou
from .box_ops import box_cxcywh_to_xyxy, box_iou, generalized_box_iou

from misc.dist import get_world_size, is_dist_available_and_initialized
from .utils import nested_tensor_from_tensor_list
class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, matcher, weight_dict, losses, alpha=0.2, gamma=2.0, eos_coef=1e-4, num_classes=80):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses 

        empty_weight = torch.ones(self.num_classes)
        empty_weight[0] = eos_coef
        self.register_buffer('empty_weight', empty_weight)

        self.alpha = alpha
        self.gamma = gamma
        self.temperature = 0.07
        
        
    def loss_align(self, outputs, targets, indices, num_boxes, log=True):
        query_image_features = outputs['query_image_features']#BS,num_queries,256//4
        support_imagee_features = outputs['support_image_features']#BS,num_classes,256//4
        normalized_query_image_features = F.normalize(query_image_features, dim=1)
        normalized_support_image_features = F.normalize(support_imagee_features, dim=1)
        logits = torch.matmul(normalized_query_image_features, normalized_support_image_features.transpose(-1,-2))/self.temperature
        positive_map = torch.zeros(logits.shape, dtype=torch.bool)
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        positive_map[idx[0],idx[1],target_classes_o.cpu()] = True
        positive_map = positive_map.to(logits.device)
        positive_logits = -logits.masked_fill(~positive_map, 0)
        negative_logits = logits  # .masked_fill(positive_map, -1000000)

        boxes_with_pos = positive_map.any(2)
        pos_term = positive_logits.sum(2)
        neg_term = negative_logits.logsumexp(2)

        nb_pos = positive_map.sum(2) + 1e-6

        box_to_token_loss = ((pos_term / nb_pos + neg_term)).masked_fill(~boxes_with_pos, 0).sum()

        tokens_with_pos = positive_map.any(1)
        pos_term = positive_logits.sum(1)
        neg_term = negative_logits.logsumexp(1)

        nb_pos = positive_map.sum(1) + 1e-6

        tokens_to_boxes_loss = ((pos_term / nb_pos + neg_term)).masked_fill(~tokens_with_pos, 0).sum()
        tot_loss = (box_to_token_loss + tokens_to_boxes_loss) / 2

        return {"loss_align": tot_loss / num_boxes}

    def loss_contrastive(self, outputs, targets, indices, num_boxes, log=True):
        query_image_features = outputs['query_image_features'].mean(1)#BS,256//4
        support_image_features = outputs['support_image_features'].mean(1)#BS,256//4
        normalized_query_image_features = F.normalize(query_image_features, dim=1)
        normalized_support_image_features = F.normalize(support_image_features, dim=1) #<- this is already normalized in data preprocessing section.
        logits = torch.matmul(normalized_query_image_features, normalized_support_image_features.t())/self.temperature
        labels = torch.arange(logits.shape[0], device=logits.device)
        loss_q = F.cross_entropy(logits, labels)
        loss_s = F.cross_entropy(logits.t(), labels)
        loss = (loss_q + loss_s) / 2.0
        return {'loss_contrastive':loss}
    
    def loss_visual_text_align_rex(self, outputs, targets, indices, num_boxes, log=True):
        support_imagee_features = outputs['support_image_features']
        support_text_features = outputs['support_avg_feat']
        
        target = outputs['support_text_labels'].to(torch.long)
        temp_scare = outputs['temp_scare']
        loss = 0
        bs = support_imagee_features.shape[0]
        for i in range(bs):
            mask =target[i]==1
            support_imagee_i = support_imagee_features[i][mask]
            supprot_text_i = support_text_features[i][mask]
            # support_imagee_i /= support_imagee_i.norm(dim=-1, keepdim=True)
            # supprot_text_i /= supprot_text_i.norm(dim=-1, keepdim=True)
            # logits = torch.matmul(support_imagee_i, supprot_text_i.t())/ self.lang_log_scale[0].exp()
            # logits = (logits + logits.t()) / 2  # 对称化处理
            support_imagee_i_normal = F.normalize(support_imagee_i, dim=1)
            supprot_text_i_normal = F.normalize(supprot_text_i, dim=1)
            logits = torch.matmul(support_imagee_i_normal, supprot_text_i_normal.t())/temp_scare
            # logits = (logits + logits.t()) / 2  # 对称化处理
            log_dig = torch.diag(logits)
            # labels = torch.arange(logits.shape[0], device=logits.device)
            
            labels_one = torch.eye(logits.shape[0], device=logits.device)
            lab_dig = torch.diag(labels_one)
            # sacre = 100
            loss_mse = F.l1_loss(log_dig.float(),lab_dig.float())
            # logits = logits*logits.shape[0]
            # loss_q = F.cross_entropy(logits, labels)
            # loss_s = F.cross_entropy(logits.t(), labels)
            # loss += (loss_q + loss_s) / 2.0
            loss += loss_mse
        return {'loss_visual_text_align_rex':loss/bs}
        

    def loss_visual_text_align(self, outputs, targets, indices, num_boxes, log=True):
        # support_imagee_features = outputs['support_image_features'][0]
        # support_text_features = outputs['support_text_features'][0]
        
        # target = outputs['support_text_labels'].to(torch.long)[0]
        # mask =target==1
        # support_imagee_bach = support_imagee_features[mask]
        # supprot_text_bach = support_text_features[mask]

        # normalized_support_imagee = F.normalize(support_imagee_bach, dim=1)
        # normalized_supprot_text = F.normalize(supprot_text_bach, dim=1)
        # logits = torch.matmul(normalized_support_imagee, normalized_supprot_text.t())
        # labels = torch.arange(logits.shape[0], device=logits.device)
        # # loss = F.cross_entropy(logits, labels)
        # logits = logits*100
        
        # loss_q = F.cross_entropy(logits, labels)
        # loss_s = F.cross_entropy(logits.t(), labels)
        # loss = (loss_q + loss_s) / 2.0

        support_imagee_features = outputs['support_image_features']
        support_text_features = outputs['support_text_features']
        
        target = outputs['support_text_labels'].to(torch.long)
        loss = 0
        bs = support_imagee_features.shape[0]
        support_image_list = []
        support_text_list = []
        for i in range(bs):
            mask =target[i]==1
            support_imagee_i = support_imagee_features[i][mask]
            support_image_list.append(support_imagee_i)
            supprot_text_i = support_text_features[i][mask]
            support_text_list.append(supprot_text_i)
        
        support_imagee_cat = torch.cat(support_image_list,dim=0)
        supprot_text_cat = torch.cat(support_text_list,dim=0)

        unique_supprot_text_cat = torch.unique(supprot_text_cat.detach(),dim=0)
        # unique_supprot_text_cat = torch.unique(supprot_text_cat,dim=0)
        # print('unique_supprot_text_cat',unique_supprot_text_cat.shape)
        unique_support_imagee_cat = unique_supprot_text_cat.new_zeros(unique_supprot_text_cat.size())

        # si_c = 0
        for i in range(unique_supprot_text_cat.shape[0]):
            mask  = torch.all(supprot_text_cat == unique_supprot_text_cat[i], dim=1)
            support_imagee_cat_indx = support_imagee_cat[mask]
            
            unique_support_imagee_cat[i] = support_imagee_cat_indx.mean(dim=0)
        
        unique_support_imagee_cat = F.normalize(unique_support_imagee_cat, dim=1)
        # unique_support_imagee_cat /= unique_support_imagee_cat.norm(dim=-1, keepdim=True)
        # normalized_supprot_text = F.normalize(unique_supprot_text_cat, dim=1)
        normalized_supprot_text = unique_supprot_text_cat
        logits = torch.matmul(unique_support_imagee_cat, normalized_supprot_text.t())
        labels = torch.arange(logits.shape[0], device=logits.device)
        # loss = F.cross_entropy(logits, labels)
        # sacre = unique_supprot_text_cat.shape[0]
        sacre = 100
        logits = logits*sacre
        
        loss_q = F.cross_entropy(logits, labels)
        loss_s = F.cross_entropy(logits.t(), labels)
        loss = (loss_q + loss_s) / 2.0
        return {'loss_visual_text_align':loss}
    
    def loss_visual_disper(self, outputs, targets, indices, num_boxes, log=True):
        support_imagee_features = outputs['support_image_features']
        support_text_features = outputs['support_image_features']
        
        target = outputs['support_text_labels'].to(torch.long)
       
        bs = support_imagee_features.shape[0]
        support_image_list = []
        support_text_list = []
        for i in range(bs):
            mask =target[i]==1
            support_imagee_i = support_imagee_features[i][mask]
            support_image_list.append(support_imagee_i)
            supprot_text_i = support_text_features[i][mask]
            support_text_list.append(supprot_text_i)
        
        support_imagee_cat = torch.cat(support_image_list,dim=0)
        supprot_text_cat = torch.cat(support_text_list,dim=0)
        normalized_support_imagee_i = F.normalize(support_imagee_cat, dim=1)
        normalized_supprot_text_i = F.normalize(supprot_text_cat, dim=1)
        logits_i = torch.matmul(normalized_support_imagee_i, normalized_supprot_text_i.t())
        labels_i = torch.arange(logits_i.shape[0], device=logits_i.device)
        loss = F.cross_entropy(logits_i, labels_i)
            
        return {'loss_visual_disper':loss}

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], 0,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight.to(src_logits.device))
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        # src_logits = outputs['pred_logits']

        # idx = self._get_src_permutation_idx(indices)
        # target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        # target_classes = torch.full(src_logits.shape[:2], self.num_classes,
        #                             dtype=torch.int64, device=src_logits.device)
        # target_classes[idx] = target_classes_o

        # target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2]+1],
        #                                     dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        # target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        # target_classes_onehot = target_classes_onehot[:,:,:-1]
        # loss_ce = sigmoid_focal_loss(src_logits[idx], target_classes_onehot[idx], num_boxes, alpha=self.alpha, gamma=2) * src_logits.shape[1]
        # losses = {'loss_ce': loss_ce}
        return losses

    def loss_labels_bce(self, outputs, targets, indices, num_boxes, log=True):
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]
        loss = F.binary_cross_entropy_with_logits(src_logits, target * 1., reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {'loss_bce': loss}

    def loss_labels_focal(self, outputs, targets, indices, num_boxes, log=True):
        assert 'pred_logits' in outputs
        #modify loss
        
        src_logits = outputs['pred_logits']
        # src_logits = F.sigmoid(src_logits)
        #mask those with -inf
        #src_logits = src_logits.masked_fill(src_logits == float("-inf"), 0)
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes+1)[..., :-1].type_as(src_logits)
        loss = torchvision.ops.sigmoid_focal_loss(src_logits, target, self.alpha, self.gamma, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes

        return {'loss_focal': loss}


    def loss_labels_vfl(self, outputs, targets, indices, num_boxes, log=True):
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)

        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
        ious = torch.diag(ious).detach()

        src_logits = outputs['pred_logits']
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target

        pred_score = F.sigmoid(src_logits).detach()
        weight = self.alpha * pred_score.pow(self.gamma) * (1 - target) + target_score
        
        loss = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {'loss_vfl': loss}

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        losses = {}

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(generalized_box_iou(
                box_cxcywh_to_xyxy(src_boxes),
                box_cxcywh_to_xyxy(target_boxes)))
        
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = F.interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,
            'bce': self.loss_labels_bce,
            'focal': self.loss_labels_focal,
            'vfl': self.loss_labels_vfl,
            'contrastive': self.loss_contrastive,
            'align':self.loss_align,
            'visual_text_align':self.loss_visual_text_align,
            'visual_text_align_rex': self.loss_visual_text_align_rex,
            'visual_disper':self.loss_visual_disper
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # self.temperature = temp
        outputs_without_aux = {k: v for k, v in outputs.items() if 'aux' not in k}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_available_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            l_dict = self.get_loss(loss, outputs, targets, indices, num_boxes)
            l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
            losses.update(l_dict)

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'contrastive':
                        continue
                    if loss == 'visual_text_align':
                        continue
                    if loss == 'visual_text_align_rex':
                        continue
                    if loss == 'visual_disper':
                        continue
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}

                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_aux_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # In case of cdn auxiliary losses. For rtdetr
        if 'dn_aux_outputs' in outputs:
            assert 'dn_meta' in outputs, ''
            indices = self.get_cdn_matched_indices(outputs['dn_meta'], targets)
            num_boxes = num_boxes * outputs['dn_meta']['dn_num_group']

            for i, aux_outputs in enumerate(outputs['dn_aux_outputs']):
                # indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'contrastive' or loss=='align':
                        continue
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}

                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_dn_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

    @staticmethod
    def get_cdn_matched_indices(dn_meta, targets):
        '''get_cdn_matched_indices
        '''
        dn_positive_idx, dn_num_group = dn_meta["dn_positive_idx"], dn_meta["dn_num_group"]
        num_gts = [len(t['labels']) for t in targets]
        device = targets[0]['labels'].device
        
        dn_match_indices = []
        for i, num_gt in enumerate(num_gts):
            if num_gt > 0:
                gt_idx = torch.arange(num_gt, dtype=torch.int64, device=device)
                gt_idx = gt_idx.tile(dn_num_group)
                assert len(dn_positive_idx[i]) == len(gt_idx)
                dn_match_indices.append((dn_positive_idx[i], gt_idx))
            else:
                dn_match_indices.append((torch.zeros(0, dtype=torch.int64, device=device), \
                    torch.zeros(0, dtype=torch.int64,  device=device)))
        
        return dn_match_indices





@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)
    #import pdb;pdb.set_trace()

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes
def dice_loss(inputs, targets, num_boxes):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_boxes



