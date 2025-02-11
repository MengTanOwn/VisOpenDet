from pycocotools.coco import COCO
import tqdm
import os
ann_path = './coco2017/annotations/lvis_v1_train.json'
folder_to_be_renamed = './dataset/lvis_maskv2s14_all'
coco = COCO(ann_path)
catIds = coco.getCatIds()
pbar = tqdm.tqdm(catIds)
for catId in pbar:
    imgIds = coco.getImgIds(catIds=catId)
    cat_name = coco.loadCats(catId)[0]['name']
    cat_folder_path = os.path.join(folder_to_be_renamed,cat_name)
    if not os.path.exists(cat_folder_path):
        os.makedirs(cat_folder_path)
    os.rename(cat_folder_path,os.path.join(folder_to_be_renamed,str(catId)))
