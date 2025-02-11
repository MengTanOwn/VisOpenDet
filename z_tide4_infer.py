import warnings
warnings.filterwarnings("error", message="To copy construct from a tensor")

import os
# 设置下载模型的缓存环境变量
os.environ['HF_HOME'] = '/mnt/localdisk/tanm/torch_home'
from model.tide import TIDE
from model.encoder11 import HybridEncoder
from model.decoder11 import TideTransformer
from config import args
from dataset.Dataset_Visual_Prompt import collate_fn_vp
import dataset.transforms as T
from PIL import Image
from model.utils import nested_tensor_from_tensor_list
from model.utils import NestedTensor
import torch
import torch.nn.functional as F
import cv2
import numpy as np
import supervision as sv

import math
import os

import random


from model.utils import NestedTensor
from misc.utils import load_weight

import clip


def set_seed(seed=42):
    
    """
    设置所有的随机数生成器的种子，确保结果的可复现性。
    :param seed: 种子值
    """
    # 设置 Python 随机模块的种子
    random.seed(seed)
    
    # 设置 NumPy 的种子
    np.random.seed(seed)
    
    # 设置 PyTorch 的种子
    torch.manual_seed(seed)
    
    # 如果使用了 CUDA，还需要为 GPU 设置种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
        # 为了确保结果的可复现性，还需要关闭 cudnn 的 benchmark 功能
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class TIDEV4():
    def __init__(self,use_clip=True) -> None:
        self.img_size = 512
        self._init_seed = set_seed(42)
        self.model = TIDE(   
            backbone=args.backbone,
            encoder=HybridEncoder(in_channels=args.backbone_dims,
                                    BMHA=args.BMHA,
                                    raw_support_feat_dim=args.support_feat_dim,
                                    dim_feedforward=args.encoder_dim_feedforward,
                                    num_fusion_layers=args.num_fusion_layers,
                                    use_mask_head=args.use_mask_head,
                                    use_text = args.use_text,
                                    use_visual_prompt=args.use_visual_prompt,
                                    max_support_len=args.max_support_len,
                                    ),                                                                            
            decoder=TideTransformer(num_classes=args.max_support_len,
                                        raw_support_feat_dim=args.support_feat_dim,
                                        feat_channels=[512, 512, 512],
                                        num_denoising=args.num_denoising,
                                        align_loss=args.use_align_loss,
                                        normalize=args.query_support_norm,
                                        num_queries=args.num_queries,
                                        mask_head=args.use_mask_head,
                                        dim_feedforward=args.decoder_dim_feedforward,
                                        learnt_init_query=args.learnt_init_query,
                                        ),                                                              
            multi_scale=None,
            l2_norm=args.query_support_norm,
            version=args.version,
            backbone_num_channels=args.backbone_dims,
            freeze_encoder=args.freeze_encoder,
            num_feature_levels=args.num_feature_level
            )
        self.model.eval()
        self.model = load_weight(args,self.model).to(args.device)
        if use_clip:
            self.clip_model,_ = clip.load("ViT-B/32", device='cuda',download_root='/mnt/localdisk/tanm/codes/tide4-tm')
            self.clip_model.eval()
    
    def load_image(self,image):
        transform = T.Compose([
                            T.ToTensor(),
                            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                    ]
                        )
        image_pillow = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        image_transformed,_= transform(image_pillow,None)
        return image_transformed

    def xywh_2_xyxy_ratio(self,box, w, h,w1,h1):
        cent_x, cent_y, box_w, box_h = box
        cent_x, cent_y, box_w, box_h = cent_x*(512/w1), cent_y*(512/h1), box_w*(512/w1), box_h*(512/h1)


        x0 = int((cent_x - box_w / 2) * w)
        y0 = int((cent_y - box_h / 2) * h)
        x1 = int((cent_x + box_w / 2) * w)
        y1 = int((cent_y + box_h / 2) * h)
        return [x0, y0, x1, y1]

    def resize_retain_hwratio(self, im):
        assert im is not None, f"Image Not Found！"
        h0, w0 = im.shape[:2]  # orig hw
        r = self.img_size / max(h0, w0)  # ratio
        if r != 1:  # if sizes are not equal
            im = cv2.resize(im, (min(math.ceil(w0 * r),self.img_size), min(math.ceil(h0 * r),self.img_size)))
        return im, (h0, w0), im.shape[:2] 

    def get_random_mapping(self,cate_list):
        BOS,EOS,mem = 0,99,[]
        random_mapping = {}
        for element in cate_list:
            rand = random.randint(BOS,EOS)
            if rand not in mem:
                mem.append(rand)
                random_mapping[element] = rand
            else:
                while rand in mem:
                    rand = random.randint(BOS,EOS)
                mem.append(rand)
                random_mapping[element] = rand
        reverse_mapping = {v:k for k,v in random_mapping.items()}
        return random_mapping,[reverse_mapping]
    
    def data_preprocess(self,prompts):
        '''
        prompts = {
            "prompt_image": ref_image,#cv format
            "type": "rect",
            "prompts": [{"category_id": 1, "rects": boxes}],
        }
        prompts = {
            "prompt_image": ref_image,#cv format
            "type": "text",
            "prompts": ["person", "car"],
        }
        '''
        prompt_type = prompts["type"]
        ref_image = prompts["prompt_image"]
        img, (h0, w0), (h, w) = self.resize_retain_hwratio(ref_image)
        query_np = np.full((self.img_size, self.img_size, 3), 0, dtype=np.uint8)
        query_np[0:h, 0:w] = img
        tag_bboxes, pos_catids = [],[]
        if prompt_type == 'rect':
            for index_ann,each in enumerate(prompts["prompts"]):
                for each_box in each['rects']:
                    xyxy = [each_box[0],each_box[1],each_box[2],each_box[3]]
                    xyxy = [int(xyxy[0]/w0*w),int(xyxy[1]/h0*h),int(xyxy[2]/w0*w),int(xyxy[3]/h0*h)]
                    tag_bboxes.append(xyxy)
                    pos_catids.append(each['category_id'])
            if tag_bboxes is not None:
                box_torch = torch.as_tensor(tag_bboxes, dtype=torch.float)
                box_torch = box_torch[None, :]
            unique_elements = list(set(pos_catids))#当前图像的类别
        else:
            text_prompts = prompts["prompts"]
            unique_elements = list(range(len(text_prompts)))#当前图像的类别

        # random_mapping = {v:v for v in unique_elements}
        BOS,EOS,mem = 0,99,[]
        random_mapping = {}
        for element in unique_elements:
            rand = random.randint(BOS,EOS)
            if rand not in mem:
                mem.append(rand)
                random_mapping[element] = rand
            else:
                while rand in mem:
                    rand = random.randint(BOS,EOS)
                mem.append(rand)
                random_mapping[element] = rand
        if prompt_type == 'rect':
            tag_labels_list = [random_mapping[element] for element in pos_catids]
            vp_dict = {'boxes': box_torch, 
                    'cates': torch.tensor(tag_labels_list),
                    'input_image_size':(query_np.shape[0],query_np.shape[1]),
                    'tag_cate_feat':[]}
        else:
            tag_cate_feat = {}
            for index,catename in enumerate(text_prompts):
                temple = f'a photo of {catename}'
                text_features = self.clip_model.encode_text(clip.tokenize([temple]).to('cuda'))
                text_features /= text_features.norm(dim=-1, keepdim=True)
                cate_feat = text_features[0].cpu().detach().numpy()
                tag_cate_feat[random_mapping[index]] = torch.from_numpy(cate_feat)
            tag_labels_list = [random_mapping[element] for element in unique_elements]
            vp_dict = {'boxes': [], 
                    'cates': torch.tensor(tag_labels_list),
                    'input_image_size':(query_np.shape[0],query_np.shape[1]),
                    'tag_cate_feat':tag_cate_feat}

        reverse_mapping = {v:k for k,v in random_mapping.items()}
        # print(prompts)
        query_img = self.load_image(query_np)
        query_img = nested_tensor_from_tensor_list([query_img])
        hw = torch.Tensor([h,w])
        sample = []
        sample.append(query_img)
        sample.append(vp_dict)
        sample.append([hw])
        sample.append([reverse_mapping])
        return tuple(sample)
    
    def post_process(self,output,reverse_mapping,h,w,h1,w1,score_t):
        predict_result = {"scores":[],"boxes":[],"labels":[]}
        logits = output["pred_logits"].cpu().sigmoid()[0]
        # logits = torch.nn.functional.softmax(output["pred_logits"][0])
        boxes = output["pred_boxes"][0]
        query_has_bbox = False
        cls_ids = []
        boxes_pred_sv = []
        scores_pred = []
        for logit, box in zip(logits, boxes):
            xywh = [x.item() for x in box.cpu()]
            xyxy = self.xywh_2_xyxy_ratio(xywh, w, h,w1,h1)
            if(xyxy[2]-xyxy[0])>w/2 or (xyxy[3]-xyxy[1])>h/2:
                # print('xyxy is too large!')
                continue
            prompt_idx = logit.argmax().item()
            score = logit[prompt_idx].item()
            if score >= score_t and prompt_idx > 0:
                cls_ids.append(prompt_idx)
                boxes_pred_sv.append(xyxy)
                scores_pred.append(score)
                query_has_bbox = True
        if query_has_bbox:
            box_sv = np.array(boxes_pred_sv)
            res_with_nms = sv.Detections(box_sv, class_id=np.array(
                cls_ids), confidence=np.array(scores_pred)).with_nms(threshold=0.3)
            for box, cls_id, score in zip(res_with_nms.xyxy, res_with_nms.class_id, res_with_nms.confidence):
                predict_result['boxes'].append(box)
                predict_result['scores'].append(score)
                predict_result['labels'].append(reverse_mapping[0].get(cls_id)) 
        return predict_result
    
    def image_process(self,target_image):
        img, (h0, w0), (h, w) = self.resize_retain_hwratio(target_image)
        query_np = np.full((self.img_size, self.img_size, 3), 0, dtype=np.uint8)
        query_np[0:h, 0:w] = img
        query_img = self.load_image(query_np)
        query_img = nested_tensor_from_tensor_list([query_img])

        return query_img,h0, w0,h, w

    def interactve_inference(self,prompts,score_t=0.2):
        #同图推理 based on box prompt
        query_img, vp_dict,hw,reverse_mapping = self.data_preprocess(prompts)
        query_img = query_img.to(args.device)
        sample_support = [{k: v.to(args.device) if k != 'input_image_size' and k != 'tag_cate_feat'  else v for k, v in vp_dict.items() }]
        output = self.model(query_img,targets=None,vp=sample_support)
        query_img_cv2 = prompts["prompt_image"]
        h,w,c = query_img_cv2.shape
        h1,w1 = hw[0]           
        return self.post_process(output,reverse_mapping,h,w,h1,w1,score_t)
    
    def get_support_feat(self,prompts,score_t=0.2):
        catBoxFeature = []
        query_img, vp_dict,hw,reverse_mapping = self.data_preprocess(prompts)
        query_img = query_img.to(args.device)
        sample_support = [{k: v.to(args.device) if k != 'input_image_size' and k != 'tag_cate_feat'  else v for k, v in vp_dict.items() }]
        output = self.model(query_img,targets=None,vp=sample_support,extract_feature_mode=True)
        bs_support,bs_support_mask = output['encoded_support'],output['support_token_mask']
        corresponding_cates = [each['cates'] for each in sample_support]
        for i in range(1):
            unique_cates = torch.unique(corresponding_cates[i])
            for cate in unique_cates:
                cate_sp_f = bs_support[i, cate]
                catBoxFeature.append([reverse_mapping[0][cate.item()],cate_sp_f.cpu().detach().numpy()])
        
        return catBoxFeature

    def generic_inference_back(self,target_image,prompts,score_t=0.2):
        #跨图推理
        query_img, vp_dict,hw,reverse_mapping = self.data_preprocess(prompts)
        query_img = query_img.to(args.device)
        sample_support = [{k: v.to(args.device) if k != 'input_image_size' and k != 'tag_cate_feat'  else v for k, v in vp_dict.items() }]
        vp_feats_dict = self.model(query_img,targets=None,vp=sample_support,extract_feature_mode=True)
        # bs_support,bs_support_mask = batch_feature['encoded_support'],batch_feature['support_token_mask']
        support_feats = NestedTensor(vp_feats_dict['encoded_support'],vp_feats_dict['support_token_mask'])
        target_query_img,h,w,h1,w1 = self.image_process(target_image)
        target_query_img = target_query_img.to(args.device)
        output = self.model(target_query_img,targets=None,y=support_feats,cross_vp=True)
        return self.post_process(output,reverse_mapping,h,w,h1,w1,score_t)

    def generic_inference(self,target_image,support_feat,reverse_mapping,score_t=0.2):
        
        target_query_img,h,w,h1,w1 = self.image_process(target_image)
        support_mask = torch.zeros((1, 100)).type_as(target_query_img.mask)
        support_feats = NestedTensor(support_feat,support_mask)
        support_feats = support_feats.to(args.device)
        target_query_img = target_query_img.to(args.device)
        output = self.model(target_query_img,targets=None,y=support_feats,cross_vp=True)
        return self.post_process(output,reverse_mapping,h,w,h1,w1,score_t)

    def text_inference(self,prompts,score_t=0.2):
        #文本推理
        query_img, vp_dict,hw,reverse_mapping = self.data_preprocess(prompts)
        query_img = query_img.to(args.device)
        sample_support = [{k: v.to(args.device) if k != 'input_image_size' 
                           and k != 'tag_cate_feat' and k != 'boxes' else v for k, v in vp_dict.items() }]
        support_feats = torch.zeros((1,100, 512), device=query_img.tensors.device)
        support_mask = torch.zeros((1, 100), device=query_img.tensors.device).type_as(query_img.mask)
        corresponding_cates = [each['cates'] for each in sample_support]
        cate_text_feat = vp_dict['tag_cate_feat']
        support_feat_list = []
        support_cate_list = []
        for i in range(1):
            unique_cates = torch.unique(corresponding_cates[i])
            for cate in unique_cates:     
                support_feats[i, cate] = cate_text_feat[cate.item()].to(torch.float32).to(args.device)
                # support_mask[i,cate] = True
                support_feat_list.append(cate_text_feat[cate.item()].to(torch.float32).to(args.device).unsqueeze(0))
                support_cate_list.append(cate.item())
        # support_mask = ~support_mask
        support_feats = NestedTensor(support_feats,support_mask)
        output = self.model(query_img,targets=None,y=support_feats,cross_vp=True)
        # boxs_900_feature = output["query_image_features"].squeeze()
        # boxs_pre = output["pred_boxes"].squeeze()
        # support_cate_list = torch.concat(support_feat_list,0)
        # support_cate_list /= support_cate_list.norm(dim=-1, keepdim=True)
        # sim = boxs_900_feature@support_cate_list.T
        # sim = sim.sigmoid().squeeze()
        # condition  = sim>0.1

        # indices = torch.where(condition)
        # values = sim[condition]
        # boxs_pre_select = boxs_pre[condition]

        # cls_ids = []
        # boxes_pred_sv = []
        # scores_pred = []
        # predict_result = {"scores":[],"boxes":[],"labels":[]}
        # query_img_cv2 = prompts["prompt_image"]
        # h,w,c = query_img_cv2.shape
        # h1,w1 = hw[0]    
        # for logit, box in zip(values, boxs_pre_select):
        #     xywh = [x.item() for x in box.cpu()]
        #     xyxy = self.xywh_2_xyxy_ratio(xywh, w, h,w1,h1)
        #     prompt_idx = 1
        #     score = logit.item()
        #     if score > score_t and prompt_idx > 0:
        #         cls_ids.append(prompt_idx)
        #         boxes_pred_sv.append(xyxy)
        #         scores_pred.append(score)
        #         query_has_bbox = True
        # if query_has_bbox:
        #     box_sv = np.array(boxes_pred_sv)
        #     res_with_nms = sv.Detections(box_sv, class_id=np.array(
        #         cls_ids), confidence=np.array(scores_pred)).with_nms(threshold=0.3)
        #     for box, cls_id, score in zip(res_with_nms.xyxy, res_with_nms.class_id, res_with_nms.confidence):
        #         predict_result['boxes'].append(box)
        #         predict_result['scores'].append(score)
        #         predict_result['labels'].append(1) 

        # return  predict_result 
        query_img_cv2 = prompts["prompt_image"]
        h,w,c = query_img_cv2.shape
        h1,w1 = hw[0]      
        return self.post_process(output,reverse_mapping,h,w,h1,w1,score_t)



        
