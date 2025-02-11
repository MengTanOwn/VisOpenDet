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
from misc import dist

from torch.nn.parallel import DistributedDataParallel



def load_image(image):
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image_pillow = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image_transformed, _ = transform(image_pillow, None)
    return image_transformed

def resize_to_closest_14x(cv2img):
    h, w = cv2img.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((224, 224, 3), dtype=np.uint8)
    h_new = int(np.ceil(h / 14) * 14)
    w_new = int(np.ceil(w / 14) * 14)
    cv2img = cv2.resize(cv2img, (w_new, h_new))
    return cv2img

@torch.no_grad()
def gen_feat_by_cate(rank, world_size, ann_path, output_path, image_folder_path, dino, poly=False):
    """
    ann_path: str, path to coco annotation file
    output_path: str, path to output folder
    image_folder_path: str, path to image folder
    dino: DINO model
    poly: bool, whether to use polygon annotation
    function: generate features for cropped images according to their bounding box annotations and save them in their corresponding coco category_id folders
    """
    coco = COCO(ann_path)
    catIds = coco.getCatIds()
    pbar = tqdm.tqdm(catIds)
    
    for catId in pbar:
        imgIds = coco.getImgIds(catIds=catId)
        cat_folder_path = os.path.join(output_path, str(catId))
        if not os.path.exists(cat_folder_path):
            os.makedirs(cat_folder_path)

        # Distribute the image IDs among processes
        imgIds = imgIds[rank::world_size]
        
        pbar2 = tqdm.tqdm(imgIds)
        for imgId in pbar2:
            img_info = coco.loadImgs(imgId)[0]
            img_path = os.path.join(image_folder_path, str(img_info['id']).zfill(12) + '.jpg')
            img = cv2.imread(img_path)
            
            if poly:
                annIds = coco.getAnnIds(imgIds=imgId, catIds=catId, iscrowd=None)
                anns = coco.loadAnns(annIds)

                for index, ann in enumerate(anns):
                    mask = coco.annToMask(ann)
                    mask = np.expand_dims(mask, axis=2)
                    mask = np.repeat(mask, 3, axis=2)
                    img_masked = img * mask
                    x, y, w, h = ann['bbox']

                    try:
                        img_masked = img_masked[int(y):int(y + h), int(x):int(x + w)]
                        img_masked = resize_to_closest_14x(img_masked)
                    except:
                        continue

                    feat = dino(load_image(img_masked).unsqueeze(0).to(device=args.device))[0]
                    feat = feat.cpu().detach().numpy()
                    np.save(os.path.join(cat_folder_path, str(ann['id'])), feat)
            else:
                annIds = coco.getAnnIds(imgIds=imgId, catIds=catId, iscrowd=None)
                anns = coco.loadAnns(annIds)

                for index, ann in enumerate(anns):
                    x, y, w, h = ann['bbox']

                    try:
                        img_cropped = img[int(y):int(y + h), int(x):int(x + w)]
                        img_cropped = resize_to_closest_14x(img_cropped)
                    except:
                        continue

                    feat = dino(load_image(img_cropped).unsqueeze(0).to(device=args.device))[0]
                    feat = feat.cpu().detach().numpy()
                    np.save(os.path.join(cat_folder_path, str(ann['id'])), feat)

if __name__ == '__main__':
    # Initialize torch distributed
    dist.init_distributed()
    
    # Get rank and world_size
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    # Use DistributedDataParallel to wrap the model
    dino = args.support_backbone.to(args.device)
    dino = dist.warp_model(args.support_backbone)

    gen_feat_by_cate(rank, world_size, args.dino_ann_path, args.dino_feats_folder_path, args.dino_image_folder_path, dino, poly=False)
