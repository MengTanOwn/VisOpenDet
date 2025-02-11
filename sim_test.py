from config import args
import numpy as np
import torch
import os
import dataset.transforms as T
from torchvision.transforms import functional as tvF
from PIL import Image
import cv2
path = os.path.join("dataset", "MaskedImages")

import clip
model,preprocess = clip.load("ViT-B/32", device='cpu')
model = args.support_backbone
def resize_2_16x16(image):
    image = cv2.resize(image,(16,16))
    return image
def resize_to_closest_14x(cv2img):
    h, w = cv2img.shape[:2]
    h_new = int(np.ceil(h / 14) * 14)
    w_new = int(np.ceil(w / 14) * 14)
    cv2img = cv2.resize(cv2img, (w_new, h_new))
    return cv2img
def load_image(image):
    transform = T.Compose([
                        T.ToTensor(),
                        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                ]
                    )
    
    image_pillow = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image_transformed,_= transform(image_pillow,None)
    return image_transformed
def test_sim(feat1,feat2):
    feat1/= feat1.norm(dim=-1, keepdim=True)
    feat2/= feat2.norm(dim=-1, keepdim=True)
    return (feat1@feat2.T).item()
if __name__ == '__main__':
    classes_list = ['1','2','3','4','5','6',"7",'8']
    avg_dict = {}
    for index,classes in enumerate(classes_list):
        avg = 0
        img_paths = os.listdir(os.path.join(path,classes))
        for i,imgs in enumerate(img_paths):
            try:
                # img1 = os.path.join(path,classes,img_paths[0])
                # img2 = os.path.join(path,classes,img_paths[i+1])
                # img3 = os.path.join(path,'7','screw_support2.jpg')
                img1 = resize_to_closest_14x(cv2.imread(os.path.join(path,classes,img_paths[0])))
                img2 = resize_to_closest_14x(cv2.imread(os.path.join(path,classes,img_paths[i+1])))
                img3 = resize_to_closest_14x(cv2.imread(os.path.join(os.path.join("dataset", "support_set"),'7','screw_support2.jpg')))
                img1_mask = torch.from_numpy(img1).unsqueeze(0).permute(0,3,1,2)
                img1_mask = img1_mask > 0
                tgt_mask = (img1_mask.shape[2]//14,img1_mask.shape[3]//14)
                img1_mask = tvF.resize(img1_mask,tgt_mask)[0][0].unsqueeze(0)
                feat1 = model.get_intermediate_layers(load_image(img1).unsqueeze(0),reshape=True, return_class_token=True,)
                feat1 = feat1[0][0][0].cpu()
                avg_patch_token = (img1_mask * feat1).flatten(1).sum(1) / img1_mask.sum()

                feat2 = model(load_image(img2).unsqueeze(0))
                feat3 = model(load_image(img3).unsqueeze(0))
                # feat1 = model.encode_image(preprocess(Image.open(img1)).unsqueeze(0))
                # feat2 = model.encode_image(preprocess(Image.open(img2)).unsqueeze(0))
                # feat3 = model.encode_image(preprocess(Image.open(img3)).unsqueeze(0))
                print("class:",classes,test_sim(feat1,feat2))
                avg+=test_sim(feat1,feat2)
            except Exception as e:
                print(e)
                avg = avg/(i)
                print("class:",classes,"avg:",avg)
                avg_dict[classes] = avg
    print(avg_dict)
        


