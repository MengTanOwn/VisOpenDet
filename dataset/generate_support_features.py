from config import args
import torch
import torch.nn as nn
import numpy as np
from pycocotools.coco import COCO
import os
from PIL import Image
import tqdm
import cv2
import dataset.transforms as T
from torchvision.transforms import functional as tvF
from misc import dist
from misc.utils import square_resize
import clip
clip_model, preprocessor = clip.load("ViT-B/32", device="cuda")
dist.init_distributed()
def load_image(image):
    transform = T.Compose([
                        T.ToTensor(),
                        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                ]
                    )
    image_pillow = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image_transformed,_= transform(image_pillow,None)
    return image_transformed
def resize_to_closest_16x(cv2img):
    h, w = cv2img.shape[:2]
    h_new = int(np.ceil(h / 16) * 16)
    w_new = int(np.ceil(w / 16) * 16)
    cv2img = cv2.resize(cv2img, (w_new, h_new))
    return cv2img
def resize_to_closest_14x(cv2img):
    h, w = cv2img.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((224,224,3),dtype=np.uint8)
    h_new = int(np.ceil(h / 14) * 14)
    w_new = int(np.ceil(w / 14) * 14)
    hw_max = max(h_new,w_new)
    cv2img = cv2.resize(cv2img, (hw_max, hw_max))
    return cv2img
@torch.no_grad()
def gen_feat_by_cate(ann_path,output_path,image_folder_path,dino,poly=False,reshape=False,masked=False):
    """
    ann_path: str, path to coco annotation file
    output_path: str, path to output folder
    image_folder_path: str, path to image folder
    dino: DINO model
    poly: bool, whether to use polygon annotation
    funciont: generate features for cropped images according to their bounding box annotations and save them in their corresponding coco category_id folders
    """
    coco = COCO(ann_path)
    catIds = coco.getCatIds()
    pbar = tqdm.tqdm(catIds)
    for catId in pbar:
        imgIds = coco.getImgIds(catIds=catId)
        catName = coco.loadCats(catId)[0]['name']
        clip_text_feat = clip_model.encode_text(clip.tokenize([f"This is not a photo of {catName}",f"This is a photo of {catName}","This is a blurry photo"]).to('cuda'))
        cat_folder_path = os.path.join(output_path,str(catId))
        if not os.path.exists(cat_folder_path):
            os.makedirs(cat_folder_path)
        elif len(os.listdir(cat_folder_path)) > 50000:
                continue
        pbar2 = tqdm.tqdm(imgIds)
        cnt = 0
        for imgId in pbar2:
            if cnt>=50000:
                break
            img_info = coco.loadImgs(imgId)[0]
            img_path = os.path.join(image_folder_path,str(img_info['file_name']))#.zfill(12)+'.jpg')
            img = cv2.imread(img_path)
            if poly:
                annIds = coco.getAnnIds(imgIds=imgId, catIds=catId, iscrowd=None)
                anns = coco.loadAnns(annIds)
                #filter ann with box area >= 100x100
                #anns = [ann for ann in anns if ann['area'] >= 50*50]
                for index,ann in enumerate(anns):
                    mask = coco.annToMask(ann)
                    mask = np.expand_dims(mask, axis=2)
                    mask = np.repeat(mask, 3, axis=2)
                    img_masked = img * mask
                    x, y, w, h = ann['bbox']
                    try:
                        img_masked = img_masked[int(y):int(y + h), int(x):int(x + w)]
                        h,w,c = img_masked.shape
                        img_masked = square_resize(img_masked,(max(h,w),max(h,w)))
                        
                        if reshape:
                            img_masked = cv2.resize(img_masked,(224,224))
                        else:
                            img_masked = resize_to_closest_16x(img_masked)
                        if index < 5:
                            cv2.imwrite(f'cropped{index}.jpg',img_masked)
                    except Exception as e:
                        print(e)
                        continue
                    if masked:
                        assert reshape!=True 
                        feat = dino.get_intermediate_layers(load_image(img_masked).unsqueeze(0).to(device=args.device), 
                                return_class_token=True, reshape=True)
                        patch_tokens = feat[0][0][0].cpu()
                        tgt_resize_shape = (img_masked.shape[0]//14,img_masked.shape[1]//14)
                        mask = tvF.resize(torch.from_numpy(mask).permute(2,0,1),size=tgt_resize_shape)[0].unsqueeze(0)
                        if mask.sum() <= 0.5:
                            continue
                        feat = (mask * patch_tokens).flatten(1).sum(1) / mask.sum()
                    else:
                        feat = dino(load_image(img_masked).unsqueeze(0).to(device=args.device))[0]
                    feat = feat.cpu().detach().numpy()
                    np.save(os.path.join(cat_folder_path, str(ann['id'])), feat)
            else:
                annIds = coco.getAnnIds(imgIds=imgId, catIds=catId, iscrowd=None)
                anns = coco.loadAnns(annIds)
                #filter ann with box area >= 100x100
                #anns = [ann for ann in anns if ann['area'] >= 10000]
                for index,ann in enumerate(anns):
                    if index>50000:
                        break
                    x, y, w, h = ann['bbox']
                    try:
                        img_cropped = img[int(y):int(y + h), int(x):int(x + w)]
                        max_hw = int(max(h,w))
                        if max_hw<88:
                            continue
                        #fill it to a square
                        img_cropped = square_resize(img_cropped,(max_hw,max_hw))    
                        if reshape or max_hw > 224:
                            img_cropped = cv2.resize(img_cropped,(224,224))
                        else:
                            img_cropped = resize_to_closest_14x(img_cropped)
                    except Exception as e:
                        print(e)
                        continue
                    clip_img = preprocessor(Image.fromarray(cv2.cvtColor(img_cropped, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device=args.device)
                    clip_img_feat = clip_model.encode_image(clip_img)
                    clip_img_feat /= clip_img_feat.norm(dim=-1, keepdim=True)
                    clip_text_feat /= clip_text_feat.norm(dim=-1, keepdim=True)
                    similarity = (100.0 * clip_img_feat @ clip_text_feat.T).softmax(dim=-1)
                    if similarity[0][1].item() < 0.55:
                        print('skipping:',similarity[0][0].item(),similarity[0][1].item(),catName)
                        continue
                    else:
                        ...
                        #print("sim:",similarity[0][0].item(),similarity[0][1].item())
                    if index < 5:
                        cv2.imwrite(f'cropped{index}.jpg',img_cropped)
                    feat = dino(load_image(img_cropped).unsqueeze(0).to(device=args.device))[0]
                    feat = feat.cpu().detach().numpy()
                    np.save(os.path.join(cat_folder_path,str(ann['id'])), feat)
                    cnt+=1
if  __name__ == '__main__'   :
    gen_feat_by_cate(args.dino_ann_path,args.dino_feats_folder_path,args.dino_image_folder_path,args.support_backbone.to(device=args.device),poly=True,reshape=True,masked=True)
