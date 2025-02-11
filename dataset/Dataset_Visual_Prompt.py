from torch.utils.data import dataset
import copy
from pycocotools.coco import COCO
try:
    import dataset.transforms as T
    from model.utils import nested_tensor_from_tensor_list
except:
    print('use local transforms!!!!!!!')
    import transforms as T
    from model.utils import nested_tensor_from_tensor_list
from PIL import Image
import os
import cv2
import numpy as np
import random
random.seed(123)
import torch
import torch.nn.functional as F
from typing import Tuple
from copy import deepcopy
import math
import pickle

# model, preprocess = clip.load('ViT-B/32',download_root='/tanm/projects/tidev3.5')
# model.eval()

class COCODatasetVisualPrompt(dataset.Dataset):
    def __init__(self,ann_path,image_folder,cate_feat=None,aug=False,strong_aug=False,train=False,support_norm=False,return_mask=False,mixed_support_selection=False,val_cls_ids=None,region_detect=False,use_text=False,bg_feat=None,use_bg=True,max_support_len=81):
        super(COCODatasetVisualPrompt, self).__init__()
        self.bg_feat = torch.load(bg_feat) if bg_feat is not None else None
        self.coco = COCO(ann_path)
        self.cats = self.coco.cats
        self.img_size = 1024
        self.cat_id2name = {cat_id: cat_name['name'] for cat_id, cat_name in self.cats.items()}
        if not use_text:
            self.coco_cate_feat = pickle.load(open(cate_feat, 'rb'))
        self.image_folder = image_folder
        self.image_ids = self.coco.getImgIds()
        self.cateid_2_dino_feat = {}
       
        self.training = train
        self.aug = aug
        self.strong_aug = strong_aug
        self.support_norm = support_norm
        self.return_mask = return_mask
        self.max_support_len = max_support_len

        self.use_text = use_text

        
    def __len__(self):
        return len(self.image_ids)
    
    # def get_text_feature(self):
    #     cat_name2features = {}
    #     with torch.no_grad():
    #         for idx,category_name in self.cat_id2name.items():
    #             cat_name2features[idx]=model.encode_text(clip.tokenize([self.text_.format(category_name)]))
    #     return cat_name2features
    def load_image(self, image,target):
        if self.aug:
            scales = [480, 512, 544, 576, 608, 640, 672, 704]
            max_size = 704
            scales2_resize = [480, 500, 600]
            transform = T.Compose(
                [
                    #T.RandomHorizontalFlip(),
                    T.RandomSelect(
                        T.RandomResize(scales, max_size=max_size),
                        T.Compose([
                            T.RandomResize(scales2_resize),
                            #T.RandomSizeCrop(*scales2_crop),
                            T.RandomResize(scales, max_size=max_size),
                        ])
                    ),
                    # T.RandomColorJitter(0.1),
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
        elif self.strong_aug:
            import dataset.sltransforms as SLT
            transform = T.Compose(
            [
                T.RandomHorizontalFlip(),
                T.RandomSelect(
                    T.RandomResize(scales, max_size=max_size),
                    T.Compose([
                        T.RandomResize(scales2_resize),
                        #T.RandomSizeCrop(*scales2_crop),
                        T.RandomResize(scales, max_size=max_size),
                    ])
                ),
                SLT.RandomSelectMulti([
                    #SLT.RandomCrop(),
                    SLT.LightingNoise(),
                    SLT.AdjustBrightness(2),
                    SLT.AdjustContrast(2),
                ]),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        else:
            transform = T.Compose([
                        T.ToTensor(),
                        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                ]
                    )
        image_pillow = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if target is not None:
            image_transformed, tgt = transform(image_pillow, target)
            return image_transformed, tgt
        else:
            image_transformed,_= transform(image_pillow,None)
            return image_transformed

    def resize_retain_hwratio(self, i):
        """Loads a single image by index, returning the image, its original dimensions, and resized dimensions."""
        image_info = self.coco.loadImgs(self.image_ids[i])[0]
        # image_path = os.path.join(self.image_folder,os.path.basename(image_info['file_name']))
        image_path = os.path.join(self.image_folder,image_info['file_name']) if image_info.get("file_name") is not None else os.path.join(self.image_folder,str(image_info['id']).zfill(12)+".jpg")
        # image_path = os.path.join(self.image_folder,str(image_info['id']).zfill(12)+".jpg")
        
        im = cv2.imread(image_path)  # BGR
        assert im is not None, f"Image Not Found {image_path}"
        h0, w0 = im.shape[:2]  # orig hw
        r = self.img_size / max(h0, w0)  # ratio
        if r != 1:  # if sizes are not equal
            im = cv2.resize(im, (min(math.ceil(w0 * r),self.img_size), min(math.ceil(h0 * r),self.img_size)))
        return im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized
    
    def clip_bbox(self,bbox, img_width, img_height):
        """
        对边界框的坐标进行裁剪，防止越界。
        
        参数:
        bbox -- 边界框坐标列表，格式为 [x, y, x+w, y+h]
        img_width -- 图像宽度
        img_height -- 图像高度
        
        返回:
        clipped_bbox -- 裁剪后的边界框坐标列表
        """
        x, y, x_max, y_max = bbox
        # if x<0 or y<0 or x_max>img_width or y_max >img_height:
        #     return False
        # 确保 x, y 坐标不会小于0
        x = max(0, x)
        y = max(0, y)
        # 确保 x_max 不会大于图片宽度
        x_max = min(img_width - 1, x_max)
        # 确保 y_max 不会大于图片高度
        y_max = min(img_height - 1, y_max)

        x_max = max(0, x_max)
        y_max = max(0, y_max)
        if x==0 and y==0 and x_max==0 and y_max==0:
            return False
        if x_max<x or y_max<y:
            return False
        
        return [x, y, x_max, y_max]
    
    def __getitem__(self, index):
        # type_rand = random.randint(0, 1)
        type_rand = 1
        if type_rand == 0 and self.training:
            indices = [index] + [random.randint(0,len(self.image_ids)-1)]  # 1 additional image indices
            s = self.img_size
            tag_bboxes, pos_catids, pos_samples, neg_samples, memory_category = [], [], [], [],[]
            copy_image_ann = []
            img2 = np.full((s * 2, s * 2, 3), 0, dtype=np.uint8)
            for i, index in enumerate(indices):
                try:
                    img, (h0, w0), (h, w) = self.resize_retain_hwratio(index)
                except:
                    return self.__getitem__(random.randint(0,len(self.image_ids)-1))
                # place img in img4
                if i == 0:  # top left
                    # base image with 4 tiles
                    img2[0:h, 0:w] = img  # img4[ymin:ymax, xmin:xmax]
                    try:
                        image_ann = self.coco.loadAnns(self.coco.getAnnIds(self.image_ids[index],iscrowd=False))
                    except:
                        image_ann = self.coco.loadAnns(self.coco.getAnnIds(self.image_ids[index]))
                    
                    if image_ann == []: return self.__getitem__(random.randint(0,len(self.image_ids)-1))
                    copy_image_ann += copy.deepcopy(image_ann)
                    for index_ann,each in enumerate(image_ann):
                        xyxy = [each['bbox'][0],each['bbox'][1],each['bbox'][0]+each['bbox'][2],each['bbox'][1]+each['bbox'][3]]
                        #if self.training == False:
                            #resacle
                        xyxy = self.clip_bbox(xyxy,w0,h0)
                        if not xyxy:
                            continue
                        xyxy = [int(xyxy[0]/w0*w/2),int(xyxy[1]/h0*h/2),int(xyxy[2]/w0*w/2),int(xyxy[3]/h0*h/2)]
                        tag_bboxes.append(xyxy)
                    
                        if 'voc' in self.image_folder.lower():
                            each["category_id"] = self.vocclsid_2_cococlsid[each["category_id"]]
                        pos_catids.append(each["category_id"])
                elif i == 1:  # top right
                    img2[s:s+h, s:s+w] = img
                    try:
                        image_ann = self.coco.loadAnns(self.coco.getAnnIds(self.image_ids[index],iscrowd=False))
                    except:
                        image_ann = self.coco.loadAnns(self.coco.getAnnIds(self.image_ids[index]))
                    
                    if image_ann == []: return self.__getitem__(random.randint(0,len(self.image_ids)-1))
                    copy_image_ann += copy.deepcopy(image_ann)
                    for index_ann,each in enumerate(image_ann):
                        xyxy = [each['bbox'][0],each['bbox'][1],each['bbox'][0]+each['bbox'][2],each['bbox'][1]+each['bbox'][3]]
                        #if self.training == False:
                            #resacle
                        xyxy = self.clip_bbox(xyxy,w0,h0)
                        if not xyxy:
                            continue
                        xyxy = [s/2+int(xyxy[0]/w0*w/2),s/2+int(xyxy[1]/h0*h/2),s/2+int(xyxy[2]/w0*w/2),s/2+int(xyxy[3]/h0*h/2)]
                        tag_bboxes.append(xyxy)
                    
                        if 'voc' in self.image_folder.lower():
                            each["category_id"] = self.vocclsid_2_cococlsid[each["category_id"]]
                        pos_catids.append(each["category_id"])

            query_np = cv2.resize(img2,(self.img_size,self.img_size))
        else:
            tag_bboxes, pos_catids, pos_samples, neg_samples, memory_category = [], [], [], [],[]
            cls2sample = {}
            try:
                image_ann = self.coco.loadAnns(self.coco.getAnnIds(self.image_ids[index],iscrowd=False))
            except:
                image_ann = self.coco.loadAnns(self.coco.getAnnIds(self.image_ids[index]))
            copy_image_ann = copy.deepcopy(image_ann)
            if image_ann == []: return self.__getitem__(random.randint(0,len(self.image_ids)-1))
            
            try:
                img, (h0, w0), (h, w) = self.resize_retain_hwratio(index)
            except:
                return self.__getitem__(random.randint(0,len(self.image_ids)-1))    
            query_np = np.full((self.img_size, self.img_size, 3), 0, dtype=np.uint8)
            query_np[0:h, 0:w] = img
            # cv2.imshow("query",query_np)
            # cv2.waitKey(0)
            #positive sample processing================================================================================================
            for index_ann,each in enumerate(image_ann):
                xyxy = [each['bbox'][0],each['bbox'][1],each['bbox'][0]+each['bbox'][2],each['bbox'][1]+each['bbox'][3]]
                
                #if self.training == False:
                    #resacle
                xyxy = self.clip_bbox(xyxy,w0,h0)
                if not xyxy:
                    continue

                xyxy = [int(xyxy[0]/w0*w),int(xyxy[1]/h0*h),int(xyxy[2]/w0*w),int(xyxy[3]/h0*h)]
                # ratio = (xyxy[2]-xyxy[0])*(xyxy[3]-xyxy[1])/(w*h)
                # if (xyxy[2]-xyxy[0])<10 or (xyxy[3]-xyxy[1])<10 or ratio>0.5 or ratio<0.01:
                #     # print('w,h:',xyxy[2]-xyxy[0],xyxy[3]-xyxy[1])
                #     continue
                tag_bboxes.append(xyxy)
                
                if 'voc' in self.image_folder.lower():
                    each["category_id"] = self.vocclsid_2_cococlsid[each["category_id"]]
                pos_catids.append(each["category_id"])
        if len(tag_bboxes) == 0:
            return self.__getitem__(random.randint(0,len(self.image_ids)-1))
        if tag_bboxes is not None:
            box_torch = torch.as_tensor(tag_bboxes, dtype=torch.float)
            box_torch = box_torch[None, :]
        unique_elements = list(set(pos_catids))#当前图像的类别

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
        reverse_mapping = {v:k for k,v in random_mapping.items()}
        if not self.use_text:
            tag_cate_feat = {random_mapping[element]:torch.from_numpy(self.coco_cate_feat[element]) for element in unique_elements}
        else:
            tag_cate_feat = {}
        tag_labels_list = [random_mapping[element] for element in pos_catids]
        
        selected_bboxes, selected_bboxes_catid = select_random_bbox_per_category(tag_bboxes,pos_catids,self.training)
        cateid2names = {random_mapping[element]:self.cat_id2name[element] for element in selected_bboxes_catid}
        selected_bboxes_catid = [random_mapping[element] for element in selected_bboxes_catid]
        vp_dict = {'boxes': torch.tensor(selected_bboxes).unsqueeze(0), 
                'cates': torch.tensor(selected_bboxes_catid),
                'cates2names':cateid2names,
                'input_image_size':(query_np.shape[0],query_np.shape[1]),
                'tag_cate_feat':tag_cate_feat,
                'vp_type':'visual'}
        # reverse_mapping = {v:k for k,v in random_mapping.items()}
        # missing_cates = list(set(tag_labels_list)-set(list(tag_labels_list_tensor.numpy())))
        # missing_cates = []
        tag_labels = torch.tensor(tag_labels_list)
        tag_bboxes = torch.Tensor(tag_bboxes)

        # tag_indexes = torch.Tensor(list(set(tag_labels_list))).type_as(tag_labels)
        # #delete the missing cates for tag_labels and tag_bboxes
        # if missing_cates != []:
        #     #delete missing categories in tag_labels_list and remove corresponding tag bboxes
        #     for cate in missing_cates:
        #         index_for_keep = torch.nonzero(tag_labels!=cate)
        #         tag_labels = tag_labels[index_for_keep].squeeze(1)
        #         tag_bboxes = tag_bboxes[index_for_keep].squeeze(1)
        target = {'boxes': tag_bboxes, 'labels': tag_labels,'size_h':h,'size_w':w}

        if self.aug and self.training:
            query_img,target= self.load_image(query_np,target) if query_np is not None else self.load_image(np.zeros((480,480,3),dtype=np.uint8),target)
            target.pop('size',None)
            target.pop('area',None)
        else:
            query_img,target = self.load_image(query_np,target) if query_np is not None else self.load_image(np.zeros((480,480,3),dtype=np.uint8),target)
        # cur_iter_pos_text = torch.zeros(num_cls_all,512)
        # if self.use_text:
        #     cur_iter_pos_text[tag_indexes] = torch.stack([self.cateid_2_clip_feat[reverse_mapping[int(key)]][0]/torch.linalg.norm(self.
        #     cateid_2_clip_feat[reverse_mapping[int(key)]][0]) for key in tag_indexes])
        
        sample = []
        sample.append(query_img)
        sample.append(vp_dict)
        sample.append(target)
        # sample.append(cur_iter_pos_text.transpose(0,1))
        #gather all the data done================================================================================================

        if self.training:
            return tuple(sample)
        else:
            if target.get('size') is not None:
                sample.append(target['size'])
            else:
                sample.append(target['hw'])
            sample.append(reverse_mapping)
            sample.append(self.image_ids[index])
            #sample.append(self.coco.dict_imgid_COCOann[imgid])
            sample.append(copy_image_ann)
            return tuple(sample)



def select_random_bbox_per_category(tag_bboxes, pos_catids,is_training=True):
    # 创建一个字典来存储每个类别的边界框
    category_bboxes = {}
    
    # 遍历所有的边界框和类别ID
    for bbox, cat_id in zip(tag_bboxes, pos_catids):
        # 如果类别ID不在字典中，则添加一个新的空列表
        if cat_id not in category_bboxes:
            category_bboxes[cat_id] = []
        # 将边界框添加到对应类别的列表中
        category_bboxes[cat_id].append(bbox)
    
    # 为每个类别随机选择多个个边界框
    selected_bboxes, selected_bboxes_catid = [],[]
    for cat_id, bboxes in category_bboxes.items():
        if bboxes:  # 确保该类别有至少一个边界框
            if is_training:
                slboxx = random.sample(bboxes,len(bboxes)-random.randint(0,len(bboxes)-1))
                for box in slboxx:
                    selected_bboxes.append(box)
                    selected_bboxes_catid.append(cat_id)
            else:
                selected_bboxes.append(random.choice(bboxes))
                selected_bboxes_catid.append(cat_id)
                # slboxx = random.sample(bboxes,len(bboxes)-random.randint(0,len(bboxes)-1))
                # for box in slboxx:
                #     selected_bboxes.append(box)
                #     selected_bboxes_catid.append(cat_id)
    
    return selected_bboxes, selected_bboxes_catid


def remapping_bach_cls(box_dict,targets):
    corresponding_cates = [each['labels'] for each in targets]
    corresponding_cates_tatal = [item.item() for sublist in corresponding_cates for item in sublist]
    unique_cates =list(set(corresponding_cates_tatal))
    map_key = {v:k for k,v in enumerate(unique_cates)}
    for i in range(len(box_dict)):
        for j in range(len(targets[i]['labels'])):
            mid_t = targets[i]['labels'][j].item()
            targets[i]['labels'][j] =torch.tensor(map_key[mid_t])
        for k in range(len(box_dict[i]['cates'])):
            box_dict[i]['cates'][k] =torch.tensor(map_key[box_dict[i]['cates'][k].item()])
        
        box_dict[i]['tag_cate_feat'] = {map_key[k]:v for k,v in box_dict[i]['tag_cate_feat'].items()}
    
        
    return box_dict,targets,map_key

def collate_fn_vp(batch):
    batch = list(zip(*batch))
    try:
        query_sample,box_dict,targets = batch
        # box_dict,targets,map_key = remapping_bach_cls(box_dict,targets)

        query_sample = nested_tensor_from_tensor_list(query_sample)
        return tuple([query_sample,box_dict,targets])
    except:
        query_sample,box_dict,targets,hw,reverse_mapping,imgid,ann = batch
        # box_dict,targets,map_key = remapping_bach_cls(box_dict,targets)
        # reverse_mapping = [{v:k for k,v in map_key.items()}]
        query_sample = nested_tensor_from_tensor_list(query_sample)
        return tuple([query_sample,box_dict,targets,hw,reverse_mapping,imgid,ann])
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    ann_path = "./annotations/instances_val2017.json"
    image_folder = "./val2017"
    train = True
    ds = COCODatasetVisualPrompt(ann_path,image_folder,train=train)
    sampler_train = torch.utils.data.RandomSampler(ds)
    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, 8, drop_last=True
    )
    data_loader = DataLoader(
        ds, batch_sampler=batch_sampler_train, collate_fn=collate_fn_vp, num_workers=0
    )
    for sample in data_loader:
        print(len(sample))
    
