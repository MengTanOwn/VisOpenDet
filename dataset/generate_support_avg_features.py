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
#from misc import dist
import clip
#dist.init_distributed()
def load_image(image):
    transform = T.Compose([
                        T.ToTensor(),
                        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                ]
                    )
    image_pillow = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image_transformed,_= transform(image_pillow,None)
    return image_transformed
def resize_to_closest_14x(cv2img):
    h, w = cv2img.shape[:2]
    h_new = int(np.ceil(h / 14) * 14)
    w_new = int(np.ceil(w / 14) * 14)
    cv2img = cv2.resize(cv2img, (w_new, h_new))
    return cv2img
def resize_to_closest_16x(cv2img):
    h, w = cv2img.shape[:2]
    h_new = int(np.ceil(h / 16) * 16)
    w_new = int(np.ceil(w / 16) * 16)
    cv2img = cv2.resize(cv2img, (w_new, h_new))
    return cv2img
def gen_masked_images_by_cate(ann_path,output_path,image_folder_path,samples_per_category=None):
    """
    ann_path: str, path to coco annotation file
    output_path: str, path to output folder
    image_folder_path: str, path to image folder
    poly: bool, whether to use polygon annotation
    funciont: generate masked images according to their bounding box annotations and save them in a folder.
    """
    print('loading coco dataset...')
    coco = COCO(ann_path)
    catIds = coco.getCatIds()
    pbar = tqdm.tqdm(catIds)
    for catindex,catId in enumerate(pbar):
        imgIds = coco.getImgIds(catIds=catId)
        cat_folder_path = os.path.join(output_path,str(catId))
        if not os.path.exists(cat_folder_path):
            os.makedirs(cat_folder_path)
        pbar2 = tqdm.tqdm(imgIds)
        
        for imgindex,imgId in enumerate(pbar2):
            if samples_per_category is not None and imgindex >= samples_per_category:
                break
            img_info = coco.loadImgs(imgId)[0]
            img_path = os.path.join(image_folder_path,str(img_info['id']).zfill(12)+'.jpg')
            img = cv2.imread(img_path)
            annIds = coco.getAnnIds(imgIds=imgId, catIds=catId)
            anns = coco.loadAnns(annIds)
            for index,ann in enumerate(anns):
                mask = coco.annToMask(ann)
                mask = np.expand_dims(mask, axis=2)
                mask = np.repeat(mask, 3, axis=2)
                img_masked = img * mask
                x, y, w, h = ann['bbox']
                img_masked = img_masked[int(y):int(y + h), int(x):int(x + w)]
                try:
                    img_masked = resize_to_closest_14x(img_masked)
                except:
                    continue
                img_masked = Image.fromarray(img_masked)
                img_masked = img_masked.convert('RGB')
                img_masked.save(os.path.join(cat_folder_path,str(imgId).zfill(12)+'_'+str(index).zfill(2)+'.jpg'))
@torch.no_grad()
def gen_feat_by_cate(ann_path,output_path,image_folder_path,support_backbone,poly=False,samples_per_category=None,text_mode=False,clip_preprocessor=None):
    """
    ann_path: str, path to coco annotation file
    output_path: str, path to output folder
    image_folder_path: str, path to image folder
    dino: DINO model
    poly: bool, whether to use polygon annotation
    funciont: generate features for cropped images according to their bounding box annotations and average them and save them in a pth.
    """
    print('loading coco dataset...')
    coco = COCO(ann_path)
    catIds = coco.getCatIds()
    pbar = tqdm.tqdm(catIds)
    if text_mode:
        #CLIP
        support_backbone.eval()
        for catId in pbar:
            cat_folder_path = os.path.join(output_path,str(catId))
            if not os.path.exists(cat_folder_path):
                os.makedirs(cat_folder_path)
            #get cat name
            cat_info = coco.loadCats(catId)[0]
            cat_name = cat_info['name']
            text_feat = support_backbone.encode_text(clip.tokenize([cat_name]).to('cuda')).float()[0]
            feat = text_feat.cpu().detach().numpy()
            feat = np.array(feat)
            feat = torch.from_numpy(feat)
            torch.save(feat,os.path.join(cat_folder_path,'avg_feats.pth'))
        exit()

    for catId in pbar:
        imgIds = coco.getImgIds(catIds=catId)
        cat_folder_path = os.path.join(output_path,str(catId))
        if not os.path.exists(cat_folder_path):
            os.makedirs(cat_folder_path)
        pbar2 = tqdm.tqdm(imgIds)
        feats = []
        for imgId in pbar2:
            img_info = coco.loadImgs(imgId)[0]
            #img_path = os.path.join(image_folder_path,str(img_info['id']).zfill(12)+'.jpg')
            img_path = os.path.join(image_folder_path,img_info['file_name'])
            img = cv2.imread(img_path)
            #print(len(feats),'/',samples_per_category)
            if samples_per_category is not None and len(feats) >= samples_per_category:
                break
            if poly:
                annIds = coco.getAnnIds(imgIds=imgId, catIds=catId,areaRng=[100*100,10e9])
                anns = coco.loadAnns(annIds)
                for index,ann in enumerate(anns):
                    mask = coco.annToMask(ann)
                    mask = np.expand_dims(mask, axis=2)
                    mask = np.repeat(mask, 3, axis=2)
                    img_masked = img * mask
                    x, y, w, h = ann['bbox']
                    img_masked = img_masked[int(y):int(y + h), int(x):int(x + w)]
                    try:
                        img_masked = resize_to_closest_14x(img_masked)
                    except:
                        continue
                    if index < 5:
                        cv2.imwrite(f'cropped{index}.jpg',img_masked)
                    feat = support_backbone(load_image(img_masked).unsqueeze(0).cuda())[0]
                    feat = feat.cpu().detach().numpy()
                    feats.append(feat)
            else:
                annIds = coco.getAnnIds(imgIds=imgId, catIds=catId)
                anns = coco.loadAnns(annIds)
                anns = [ann for ann in anns if ann['area'] >= 10000]
                for index,ann in enumerate(anns):
                    x, y, w, h = ann['bbox']
                    try:
                        img_cropped = img[int(y):int(y + h), int(x):int(x + w)]
                        if max(img_cropped.shape)>640:
                            img_cropped = cv2.resize(img_cropped,(640,640))
                        img_cropped = resize_to_closest_14x(img_cropped)
                    except Exception as e:
                        print(e)
                        continue
                    if index < 5:
                        cv2.imwrite(f'cropped{index}.jpg',img_cropped)
                    feat = support_backbone(load_image(img_cropped).unsqueeze(0).cuda())[0]
                    feat = feat.cpu().detach().numpy()
                    feats.append(feat)
        feats = np.array(feats)
        feats = np.mean(feats,axis=0)
        feats = torch.from_numpy(feats)
        torch.save(feats,os.path.join(cat_folder_path,'avg_feats.pth'))
        
    
if  __name__ == '__main__'   :
    #gen_masked_images_by_cate(args.dino_ann_path,args.mask_image_folder,args.dino_image_folder_path,samples_per_category=10)
    gen_feat_by_cate(args.dino_ann_path,args.dino_feats_folder_path,args.dino_image_folder_path,args.support_backbone,poly=False,samples_per_category=100)
