import warnings
warnings.filterwarnings("error", message="To copy construct from a tensor")

from model.VisOpenDet import VisOpenDet
from .encoder import HybridEncoder
from .decoder import TideTransformer
from config import args
import dataset.transforms as T
from PIL import Image
from model.utils import nested_tensor_from_tensor_list
import torch
import math
import cv2
import numpy as np
import random
import supervision as sv
from misc.utils import load_weight


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class VisInference():
    def __init__(self) -> None:
        self.img_size = 1024
        self._init_seed = set_seed(42)
        self.model = VisOpenDet(   
            backbone=args.backbone,
                encoder=HybridEncoder(in_channels=args.backbone_dims,
                                    raw_support_feat_dim=args.support_feat_dim,
                                    hidden_dim =args.support_feat_dim,
                                    dim_feedforward=args.encoder_dim_feedforward, 
                                    max_support_len=args.max_support_len,
                                    ),                                                                            
                decoder=TideTransformer(num_classes=args.max_support_len,
                                        raw_support_feat_dim=args.support_feat_dim,
                                        hidden_dim =args.support_feat_dim,
                                        feat_channels=[args.support_feat_dim, args.support_feat_dim, args.support_feat_dim],
                                        normalize=args.query_support_norm,
                                        num_queries=args.num_queries,
                                        dim_feedforward=args.decoder_dim_feedforward,
                                        ),                                                                
            multi_scale=None,
            l2_norm=args.query_support_norm,
            backbone_num_channels=args.backbone_dims,
            freeze_encoder=args.freeze_encoder,
            num_feature_levels=args.num_feature_level
            )
        self.model.eval()
        self.model = load_weight(args,self.model).to(args.device)
    
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
        cent_x, cent_y, box_w, box_h = cent_x*(self.img_size/w1), cent_y*(self.img_size/h1), box_w*(self.img_size/w1), box_h*(self.img_size/h1)
        x0 = int((cent_x - box_w / 2) * w)
        y0 = int((cent_y - box_h / 2) * h)
        x1 = int((cent_x + box_w / 2) * w)
        y1 = int((cent_y + box_h / 2) * h)
        return [x0, y0, x1, y1]

    def resize_retain_hwratio(self, im):
        assert im is not None, f"Image Not Found！"
        h0, w0 = im.shape[:2]  # orig hw
        im = cv2.resize(im, (self.img_size, self.img_size))
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
            "prompts": [{"category_id": 1, "rects": boxes}],
        }
        '''
        ref_image = prompts["prompt_image"]
        img, (h0, w0), (h, w) = self.resize_retain_hwratio(ref_image)
        query_np = np.full((self.img_size, self.img_size, 3), 0, dtype=np.uint8)
        query_np[0:h, 0:w] = img
        tag_bboxes, pos_catids = [],[]
        
        for _,each in enumerate(prompts["prompts"]):
            for each_box in each['rects']:
                xyxy = [each_box[0],each_box[1],each_box[2],each_box[3]]
                xyxy = [int(xyxy[0]/w0*w),int(xyxy[1]/h0*h),int(xyxy[2]/w0*w),int(xyxy[3]/h0*h)]
                tag_bboxes.append(xyxy)
                pos_catids.append(each['category_id'])
        if tag_bboxes is not None:
            box_torch = torch.as_tensor(tag_bboxes, dtype=torch.float)
            box_torch = box_torch[None, :]
        unique_elements = list(set(pos_catids))#当前图像的类别
        
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
        
        tag_labels_list = [random_mapping[element] for element in pos_catids]
        vp_dict = {'boxes': box_torch, 
                'cates': torch.tensor(tag_labels_list),
                'input_image_size':(query_np.shape[0],query_np.shape[1]),
                }
        reverse_mapping = {v:k for k,v in random_mapping.items()}
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
        boxes = output["pred_boxes"][0]
        query_has_bbox = False
        cls_ids = []
        boxes_pred_sv = []
        scores_pred = []
        for logit, box in zip(logits, boxes):
            xywh = [x.item() for x in box.cpu()]
            xyxy = self.xywh_2_xyxy_ratio(xywh, w, h,w1,h1)
            if(xyxy[2]-xyxy[0])>w/2 or (xyxy[3]-xyxy[1])>h/2:
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
                predict_result['scores'].append(math.sqrt(score))
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
        #interactive prompt  based on box prompt
        query_img, vp_dict,hw,reverse_mapping = self.data_preprocess(prompts)
        query_img = query_img.to(args.device)
        sample_support = [{k: v.to(args.device) if k != 'input_image_size' and k != 'tag_cate_feat'  else v for k, v in vp_dict.items() }]
        output = self.model(query_img,targets=None,vp=sample_support)
        query_img_cv2 = prompts["prompt_image"]
        h,w,_ = query_img_cv2.shape
        h1,w1 = hw[0]           
        return self.post_process(output,reverse_mapping,h,w,h1,w1,score_t)



        
