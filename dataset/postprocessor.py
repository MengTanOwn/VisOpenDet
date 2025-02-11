import torch
import cv2
import time
import numpy as np
import supervision as sv
from config import args
from torchvision.transforms import functional as tvF
import dataset.transforms as T
from PIL import Image
from thop import profile,clever_format
from model.utils import nested_tensor_from_tensor_list
import os
from misc.utils import square_resize
import matplotlib.pyplot as plt
import torch.nn.functional as F
import clip
import torch.nn as nn
activation = {}
gradient = {}
def get_activation(name):
    def hook(model, input, output):
        if name not in activation.keys():
            activation[name] = output
        else:
            activation[name]=torch.cat([output,activation[name]],dim=0)
    return hook
def get_gradient(name):
    def hook(model, input, output):
        if name not in gradient.keys():
            gradient[name] = input
        else:
            gradient[name] = torch.cat([input,gradient[name]],dim=0)
    return hook
box_annotator = sv.BoxAnnotator()
mask_annotator = sv.MaskAnnotator()
def resize_to_closest_14x(cv2img):
    h, w = cv2img.shape[:2]
    h_new = int(np.ceil(h / 14) * 14)
    w_new = int(np.ceil(w / 14) * 14)
    cv2img = cv2.resize(cv2img, (w_new, h_new))
    return cv2img
def xywh_2_xyxy(box, w, h):
    cent_x, cent_y, box_w, box_h = box
    x0 = int((cent_x - box_w / 2) * w)
    y0 = int((cent_y - box_h / 2) * h)
    x1 = int((cent_x + box_w / 2) * w)
    y1 = int((cent_y + box_h / 2) * h)
    return [x0, y0, x1, y1]
@torch.no_grad()
def load_image(image):
    transform = T.Compose([
                        T.ToTensor(),
                        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                ]
                    )
    image_pillow = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image_transformed,_= transform(image_pillow,None)
    return image_transformed
#@torch.no_grad()
def process(model,backbone_support,query_img_path,support_set,resize_shape=None,POS_THRES=0.5,iou=None,classes:str = None,use_fixed=False,fixed_classes=81,debug=False,masked_feats=False,use_square_resize=False,square_resize_value=(0,0,0),text=None):
    global query_img_CV2
    assert square_resize_value == (0,0,0) or (255,255,255)
    if use_fixed:
        support_list = [torch.ones(args.support_feat_dim).to(args.device)/torch.linalg.norm(torch.ones(args.support_feat_dim)).to(args.device)]
        classes = os.listdir(args.dino_feats_folder_path)
        classes.sort(key=lambda x:int(x))
        classes.insert(0,'background')
    elif args.bg_feat is not None:
        support_list = [torch.load(args.bg_feat).to(args.device)/torch.linalg.norm(torch.load(args.bg_feat).to(args.device))]
    else:
        if args.support_norm:
            support_list = [torch.ones(args.support_feat_dim).to(args.device)/torch.linalg.norm(torch.ones(args.support_feat_dim)).to(args.device)]
        else:
            support_list = [torch.ones(args.support_feat_dim).to(args.device)]
        if args.use_bg == False:
            support_list = []
    query_img_CV2 = cv2.imread(query_img_path) if not resize_shape else cv2.resize(cv2.imread(query_img_path),resize_shape)
    if len(args.backbone_dims) == 1:
        query_img_CV2 = resize_to_closest_14x(query_img_CV2)
    h,w,c = query_img_CV2.shape
    query_img_PIL = load_image(query_img_CV2)
    if classes is not None:
        if use_fixed:
            temp = os.listdir(args.dino_feats_folder_path)
            temp.sort(key=lambda x:int(x))
            #temp = temp[:3]
            for i in temp:
                try:
                    feat = os.path.join(args.dino_feats_folder_path,str(i),'avg_feats.pth')
                    feat = torch.load(feat).to(args.device)
                except Exception as e:
                    print(e)
                    continue
                feat = feat/torch.linalg.norm(feat) if args.support_norm else feat
                support_list.append(feat)
            support_samples = torch.stack(support_list)
        else:
            if debug:
                import pdb;pdb.set_trace()
                support_list_2 = [torch.ones(args.support_feat_dim).to(args.device)/torch.linalg.norm(torch.ones(args.support_feat_dim)).to(args.device)]
                for i in range (1,fixed_classes):
                    try:
                        feat = os.path.join(args.dino_feats_folder_path,str(i),'avg_feats.pth')
                        feat = torch.load(feat).to(args.device)
                    except:
                        continue
                    feat = feat/torch.linalg.norm(feat) if args.support_norm else feat
                    support_list_2.append(feat)
                support_samples_fixed = torch.stack(support_list_2)
            for cls in support_set:
                shots_list = []
                for img_path in support_set[cls]:
                    support_img = cv2.imread(img_path)
                    if use_square_resize:
                        hs,ws,c = support_img.shape
                        if hs>224 or ws>224:
                            support_img = square_resize(support_img,(224,224),square_resize_value)
                        else:
                            maxhw = int(max(hs,ws))
                            support_img = square_resize(support_img,(maxhw,maxhw),square_resize_value)
                            support_img = resize_to_closest_14x(support_img)
                    else:
                        support_img = resize_to_closest_14x(support_img)
                    support_img_mask = torch.from_numpy(support_img).unsqueeze(0).permute(0,3,1,2)
                    support_img_mask = (support_img_mask > 0)
                    tgt_mask_shape = (support_img_mask.shape[2]//14,support_img_mask.shape[3]//14)
                    support_img_mask = tvF.resize(support_img_mask,tgt_mask_shape)[0][0].unsqueeze(0)
                    if support_img_mask.sum() > (~support_img_mask).sum() and masked_feats:
                        temp = backbone_support.get_intermediate_layers(load_image(support_img).unsqueeze(0).to(args.device),reshape=True, return_class_token=True,)
                        patch_tokens = temp[0][0][0].cpu()
                        temp = (support_img_mask * patch_tokens).flatten(1).sum(1) / support_img_mask.sum()
                    else:
                        if hasattr(backbone_support,'encode_image'):
                            temp = backbone_support.encode_image(args.clip_preprocessor(Image.open(img_path)).unsqueeze(0).to(args.device))[0]
                        else:
                            temp = backbone_support(load_image(support_img).unsqueeze(0).to(args.device))[0]
                    temp = temp.cpu().detach().numpy()
                    shots_list.append(temp)
                shots_list = np.array(shots_list)
                shots_list = np.mean(shots_list,axis=0)
                avg_shots = torch.from_numpy(shots_list).to(device=args.device)
                avg_shots = avg_shots/torch.linalg.norm(avg_shots) if args.support_norm else avg_shots
                #avg_shots = torch.mean(torch.stack(shots_list),dim=0) if len(shots_list)>1 else shots_list[0]
                support_list.append(avg_shots)
            support_samples = torch.stack(support_list)
            #pad to args.max_support_len with (384) 0
            # if len(support_samples) < args.max_support_len:
            #     support_samples = torch.cat([support_samples,torch.zeros(args.max_support_len-len(support_samples),args.support_feat_dim).to(args.device)])
            
    if text is not None:
        clip_model = args.support_backbone.to(args.device)
        text_features = [torch.ones(args.support_feat_dim).to(args.device)/torch.linalg.norm(torch.ones(args.support_feat_dim)).to(args.device)]
        for txt in text:
            text_features.append(clip_model.encode_text(clip.tokenize(txt).to(args.device))[0])
        text_feats = torch.stack(text_features)
    if classes is None:
        support_samples = text_feats
        text_feats=None    
    else:
        text_feats = None
    query_sample = nested_tensor_from_tensor_list([query_img_PIL]).to(device=args.device)
    support_samples = nested_tensor_from_tensor_list(support_samples.unsqueeze(0).transpose(1,2)).to(device=args.device)
    text_samples = nested_tensor_from_tensor_list(text_feats.unsqueeze(0).transpose(1,2)).to(device=args.device) if text_feats is not None else None
    if classes is None:
        classes = ['background']+text
    post_process(model,h,w,query_sample,support_samples,POS_THRES=POS_THRES,iou=iou,classes=classes,query_img_path=query_img_path,text_feats=text_samples)

@torch.no_grad()
def post_process(model,h,w,query_sample,support_samples,POS_THRES=0.5,iou=None,classes:str = None,query_img_path=None,text_feats=None):
    inference_time_start = time.time()
    # model.encoder.input_proj[-1].register_forward_hook(get_activation('fusion'))
    # model.encoder.input_proj[-1].register_backward_hook(get_gradient('fusion'))
    # model.decoder.input_proj[0].register_forward_hook(get_activation('fusion1'))
    # model.decoder.input_proj[0].register_backward_hook(get_gradient('fusion1'))
    # model.decoder.input_proj[1].register_forward_hook(get_activation('fusion2'))
    # model.decoder.input_proj[1].register_backward_hook(get_gradient('fusion2'))
    # model.decoder.input_proj[2].register_forward_hook(get_activation('fusion3'))
    # model.decoder.input_proj[2].register_backward_hook(get_gradient('fusion3'))
    # model.encoder.input_proj[0].register_forward_hook(get_activation('fusion4'))
    # model.encoder.input_proj[0].register_backward_hook(get_gradient('fusion4'))
    # model.encoder.input_proj[1].register_forward_hook(get_activation('fusion5'))
    # model.encoder.input_proj[1].register_backward_hook(get_gradient('fusion5'))
    # model.encoder.input_proj[2].register_forward_hook(get_activation('fusion6'))
    # model.encoder.input_proj[2].register_backward_hook(get_gradient('fusion6'))

    
    
    outputs = model(query_sample, support_samples,None,text_feats)
    
    inference_time = time.time()-inference_time_start
    print("inference time:",inference_time)
    flops, params = profile(model, inputs=(query_sample, support_samples,))
    flops, params = clever_format([flops, params], "%.3f")
    print(flops, params)
    if 'focal' in args.resume:
        logits = outputs["pred_logits"].cpu().sigmoid()[0]
    else:
        logits = torch.nn.functional.softmax(outputs["pred_logits"][0])#.cpu()
    boxes = outputs["pred_boxes"][0]#.cpu()
 
    sv_masks = []
    sv_boxes = []
    sv_classes = []
    sv_scores = []
    backward_data = []
    if args.use_mask_head:
        masks = F.interpolate(outputs["pred_masks"].squeeze(2),(h,w),mode='bilinear',align_corners=False)
        masks = (masks[0].sigmoid()>0.1).cpu()
    else:
        masks = [0]*len(logits)
        sv_masks = []
    for logit, box,mask in zip(logits, boxes,masks):
        xywh = [x.item() for x in box.cpu()]
        box = xywh_2_xyxy(xywh, w, h)
        prompt_cls = logit.argmax().item()
        score = logit[prompt_cls].item()
        
        if prompt_cls > 0 and score > POS_THRES:
            sv_masks.append(mask)
            backward_data.append(logit[prompt_cls].unsqueeze(0))
            sv_boxes.append(box)
            sv_classes.append(prompt_cls)
            sv_scores.append(score)
            
            print("class:", prompt_cls, "score:", score)
    # backward_data_mean = torch.cat(backward_data).mean()
    # backward_data_mean.backward()
    
    if sv_boxes:
        box_sv = np.array(sv_boxes)

        if iou and args.use_mask_head:
            res = sv.Detections(
                box_sv, mask = np.array(sv_masks),class_id=np.array(sv_classes), confidence=np.array(sv_scores)
            ).with_nms(threshold=iou,class_agnostic=False)
            masked_img = mask_annotator.annotate(
                scene=query_img_CV2,
                detections=res,
            )
            cv2.imwrite('masked.jpg',masked_img)
        elif args.use_mask_head:
            res = sv.Detections(
                box_sv, mask = np.array(sv_masks),class_id=np.array(sv_classes), confidence=np.array(sv_scores)
            )
            masked_img = mask_annotator.annotate(
                scene=query_img_CV2,
                detections=res,
            )
            cv2.imwrite('masked.jpg',masked_img)
        elif iou:
            res = sv.Detections(
                box_sv,class_id=np.array(sv_classes), confidence=np.array(sv_scores)
            ).with_nms(threshold=iou,class_agnostic=False)
        else:
            res = sv.Detections(
                box_sv,class_id=np.array(sv_classes), confidence=np.array(sv_scores)
            )
        try:
            labels = [
                f"{classes[class_id]} {confidence:0.2f}"
                for _, _, confidence, class_id, _ in res
            ]
            copy_query = query_img_CV2.copy()
            box_annotator.annotate(
                scene=query_img_CV2,
                detections=res,
                labels=labels,
            )
            masked_img = mask_annotator.annotate(
                scene=copy_query,
                detections=res,
            )
            cv2.imwrite('masked.jpg',masked_img)

        except Exception as e:
            print(e)
            pass
        if os.name == 'nt':
            cv2.imshow("query", query_img_CV2)
            cv2.waitKey(0)
        else:
            cv2.imwrite('res.jpg',query_img_CV2)
            
        exit()
        gradients1 = gradient['fusion1'][0]
        gradients2 = gradient['fusion2'][0]
        gradients3 = gradient['fusion3'][0]
        gradients4 = gradient['fusion4'][0]
        gradients5 = gradient['fusion5'][0]
        gradients6 = gradient['fusion6'][0]

        heatmap1 = torch.mean(F.tanh(gradients1), dim=1)[0]
        heatmap2 = torch.mean(F.tanh(gradients2), dim=1)[0]
        heatmap3 = torch.mean(F.tanh(gradients3), dim=1)[0]
        heatmap4 = torch.mean(F.tanh(gradients4), dim=1)[0]
        heatmap5 = torch.mean(F.tanh(gradients5), dim=1)[0]
        heatmap6 = torch.mean(F.tanh(gradients6), dim=1)[0]

        # 使用Relu函数作用于热力图
        heatmap1 = F.relu(heatmap1)
        heatmap2 = F.relu(heatmap2)
        heatmap3 = F.relu(heatmap3)
        heatmap4 = F.relu(heatmap4)
        heatmap5 = F.relu(heatmap5)
        heatmap6 = F.relu(heatmap6)

        # 对热力图进行标准化
        heatmap1 /= torch.max(heatmap1)
        heatmap1 = heatmap1.cpu().detach().numpy()
        heatmap2 /= torch.max(heatmap2)
        heatmap2 = heatmap2.cpu().detach().numpy()
        heatmap3 /= torch.max(heatmap3)
        heatmap3 = heatmap3.cpu().detach().numpy()
        heatmap4 /= torch.max(heatmap4)
        heatmap4 = heatmap4.cpu().detach().numpy()
        heatmap5 /= torch.max(heatmap5)
        heatmap5 = heatmap5.cpu().detach().numpy()
        heatmap6 /= torch.max(heatmap6)
        heatmap6 = heatmap6.cpu().detach().numpy()



        #gradients = F.relu(gradient['fusion'][0])
        #print(gradients.shape)
        #mean_gradients = torch.mean(gradients, dim=[0,2,3])
        #activations = activation['fusion']
        #print(activations.shape)
        #for i in range(len(mean_gradients)):
        #    activations[:,i,:,:] *= mean_gradients[i]
        #heatmap = torch.mean(activations,dim=1,keepdim=True)
        #print(heatmap.shape)
        #heatmap = F.relu(heatmap)
        #heatmap /= torch.max(heatmap)
        w,h = query_img_CV2.shape[:2]
        #heatmap = cv2.resize(heatmap4, (w, h))+cv2.resize(heatmap5, (w, h))+cv2.resize(heatmap6, (w, h))
        #heatmap = cv2.resize(heatmap1, (w, h))+cv2.resize(heatmap2, (w, h))+cv2.resize(heatmap3, (w, h))
        heatmap = cv2.resize(heatmap1, (w, h))+cv2.resize(heatmap2, (w, h))+cv2.resize(heatmap3, (w, h))+cv2.resize(heatmap4, (w, h))+cv2.resize(heatmap5, (w, h))+cv2.resize(heatmap6, (w, h))
        heatmap = np.uint8(255 * heatmap)
        print(heatmap.shape)
        img = cv2.resize(cv2.imread(query_img_path),(w,h))
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        grad_cam_img = heatmap * 0.2 + img
        #grad_cam_img = heatmap
        grad_cam_img = grad_cam_img / grad_cam_img.max()
        b,g,r = cv2.split(grad_cam_img)
        grad_cam_img = cv2.merge([r,g,b])
        print(grad_cam_img.shape)
        plt.imsave('heatmap.png',grad_cam_img)

