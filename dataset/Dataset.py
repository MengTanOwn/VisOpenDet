from torch.utils.data import dataset
from pycocotools.coco import COCO
import pycocotools.mask as coco_mask
try:
    import dataset.transforms as T
    from model.utils import nested_tensor_from_tensor_list
except:
    import transforms as T
    from model.utils import nested_tensor_from_tensor_list
import clip

from PIL import Image
import itertools
import os
import cv2
import bisect
import numpy as np
import random
import torch
model, preprocess = clip.load('ViT-B/32',device='cpu')
def find_closest_to_14x(num):
    return int(np.ceil(num / 14) * 14)
class COCODataset(dataset.Dataset):
    def __init__(self,ann_path,image_folder,dino_feats_folder=None,support_feat_dim=384,max_support_len=81,extra_shots=0,aug=False,strong_aug=False,train=False,support_norm=False,return_mask=False,mixed_support_selection=False,val_cls_ids=None,region_detect=False,use_text=False,use_bg=True,bg_feat=None):
        super(COCODataset, self).__init__()
        self.bg_feat = torch.load(bg_feat) if bg_feat is not None else None
        self.coco = COCO(ann_path)
        self.image_folder = image_folder
        self.image_ids = self.coco.getImgIds()
        self.cateid_2_dino_ann = {}
        self.dino_feats_folder = dino_feats_folder
        self.extra_shots = extra_shots
        self.max_support_len = max_support_len
        self.support_feat_dim = support_feat_dim
        self.training = train
        self.aug = aug
        self.strong_aug = strong_aug
        self.support_norm = support_norm
        self.return_mask = return_mask
        self.mixed_support_selection = mixed_support_selection
        self.region_detect = region_detect
        self.category_text_features = {}
        #self.text_background_feat = model.encode_text(clip.tokenize(["background"]))[0]
        self.use_text = use_text
        self.use_bg = use_bg
        if self.dino_feats_folder:
            for cates in os.listdir(self.dino_feats_folder):
                self.cateid_2_dino_ann[cates] = os.listdir(os.path.join(self.dino_feats_folder,cates))
                #remove folder with no dino features
                if len(self.cateid_2_dino_ann[cates]) == 0:
                    self.cateid_2_dino_ann.pop(cates)
                    #remove folder with no dino features using os
                    os.rmdir(os.path.join(self.dino_feats_folder,cates))
                elif len(self.cateid_2_dino_ann[cates]) < 20:
                    self.cateid_2_dino_ann.pop(cates)
            self.available_cates = list(self.cateid_2_dino_ann.keys())
            if val_cls_ids is not None:
                #remove val_cls_ids from available cates
                self.available_cates = list(set(self.available_cates)-set(val_cls_ids))
            print("available cates:",len(self.available_cates))
        else:
            self.available_cates = [range(1,81)]
            cate = self.coco.getCatIds()
            self.cateid_2_dino_ann = {str(each):[str(each)+".npy"] for each in cate}
    def __len__(self):
        return len(self.image_ids)
    def load_image(self, image,target):
        if self.aug:
            scales = [480, 512, 544, 576, 608, 640]
            max_size = 640
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

    def __getitem__(self, index):
        tag_bboxes, pos_catids, pos_samples, neg_samples, cls2sample = [], [], [], [],{}
        try:
            image_ann = self.coco.loadAnns(self.coco.getAnnIds(self.image_ids[index],iscrowd=False,areaRng=[48*48,float('inf')]))
        except:
            image_ann = self.coco.loadAnns(self.coco.getAnnIds(self.image_ids[index]))
        if image_ann == []: return self.__getitem__(random.randint(0,self.__len__()-1))
        image_info = self.coco.loadImgs(self.image_ids[index])[0]
        image_path = os.path.join(self.image_folder,image_info['file_name']) if image_info.get("file_name") is not None else os.path.join(self.image_folder,str(image_info['id']).zfill(12)+".jpg")
        query_np = cv2.imread(image_path)
        if query_np is None and self.training:
            return self.__getitem__(random.randint(0,len(self.image_ids)-1))
        if self.return_mask:
            segmentations = []
        #positive sample processing================================================================================================
        if self.region_detect:
            pos_region_samples = []
        for index,each in enumerate(image_ann):
            no_extra_shots_flag = False
            if each.get('category_id') not in cls2sample:
                cls2sample[each['category_id']] = []
            if str(each.get('category_id')) not in self.available_cates and self.training:
                no_extra_shots_flag = True
                continue
            category_name = self.coco.loadCats(each['category_id'])[0]['name']
            with torch.no_grad():
                if each['category_id'] not in self.category_text_features and self.use_text:
                    self.category_text_features[each['category_id']] = model.encode_text(clip.tokenize([category_name]))
            xyxy = [each['bbox'][0],each['bbox'][1],each['bbox'][0]+each['bbox'][2],each['bbox'][1]+each['bbox'][3]]
            tag_bboxes.append(xyxy)
            pos_catids.append(each["category_id"])
            if self.return_mask:
                segmentations.append(each['segmentation'])
                masks = convert_coco_poly_to_mask(segmentations, query_np.shape[0], query_np.shape[1])
            
            if self.mixed_support_selection:
                if random.random() < 0.5 and str(each["id"])+".npy" in self.cateid_2_dino_ann[str(each['category_id'])]:
                    pos_ann = str(each['id'])+'.npy'
                    no_extra_shots_flag = True
                else:
                    pos_ann_index = random.randint(0,len(self.cateid_2_dino_ann[str(each['category_id'])])-1)
                    pos_ann = self.cateid_2_dino_ann.get(str(each['category_id']))[pos_ann_index]
            elif self.region_detect:
                pos_ann = str(each['id'])+'.npy'
            else:
                pos_ann_index = random.randint(0,len(self.cateid_2_dino_ann[str(each['category_id'])])-1)
                pos_ann = self.cateid_2_dino_ann.get(str(each['category_id']))[pos_ann_index]
            pos_npy_feat = np.load(os.path.join(self.dino_feats_folder,str(each['category_id']),str(pos_ann))) if self.dino_feats_folder is not None else None
            pos_sample = torch.from_numpy(pos_npy_feat) if pos_npy_feat is not None else torch.zeros(self.support_feat_dim)
            if self.support_norm:
                pos_sample = pos_sample/torch.linalg.norm(pos_sample)
            if self.region_detect:
                pos_region_samples.append(pos_sample)
            elif not no_extra_shots_flag:
                cls2sample[each["category_id"]].append(pos_sample)
            elif cls2sample[each["category_id"]] == []:
                cls2sample[each["category_id"]].append(pos_sample)
            
        if self.extra_shots and not no_extra_shots_flag:
            iters_for_shots = random.randint(1,self.extra_shots)
            for _ in range(iters_for_shots):
                extra_ann_index = random.randint(0,len(self.cateid_2_dino_ann[str(each['category_id'])])-1)
                extra_ann = self.cateid_2_dino_ann[str(each['category_id'])][extra_ann_index]
                extra_npy_feat = np.load(os.path.join(self.dino_feats_folder,str(each['category_id']),str(extra_ann)))
                extra_sample = torch.from_numpy(extra_npy_feat)
                if self.support_norm:
                    extra_sample = extra_sample/torch.linalg.norm(extra_sample)
                cls2sample[each["category_id"]].append(extra_sample)
        #positive sample processing done===========================================================================================
        if len(pos_catids) == 0:return self.__getitem__(random.randint(0,len(self.image_ids)-1))        
        #negative sample processing================================================================================================
        if self.region_detect:
            num_cls_neg = random.randint(1, self.max_support_len - len(tag_bboxes))
        elif random.random() < 0.5:
            num_cls_neg = random.randint(1, self.max_support_len - len(set(pos_catids)))
        else:
            num_cls_neg = self.max_support_len - len(set(pos_catids))
        neg_cls_mem = []
        for _ in range(num_cls_neg):
            neg_cls = random.choice(list(set(self.cateid_2_dino_ann.keys())-set(pos_catids)-set(neg_cls_mem)))
            neg_cls_mem.append(neg_cls)
            try:neg_ann_index = random.randint(0,len(self.cateid_2_dino_ann[str(neg_cls)])-1)
            except:neg_ann_index = random.randint(0,len(self.cateid_2_dino_ann[neg_cls])-1)

            neg_ann = self.cateid_2_dino_ann[str(neg_cls)][neg_ann_index]

            neg_npy_feat = np.load(os.path.join(self.dino_feats_folder,str(neg_cls),str(neg_ann))) if self.dino_feats_folder is not None else None
            neg_sample = torch.from_numpy(neg_npy_feat) if neg_npy_feat is not None else torch.zeros(self.support_feat_dim)
            neg_sample_list =[]
            if self.support_norm:
                neg_sample = neg_sample/torch.linalg.norm(neg_sample)
                neg_sample_list.append(neg_sample)
            if self.extra_shots !=0:
                iters_for_shots = random.randint(1,self.extra_shots)
                for _ in range(iters_for_shots):
                    extra_ann_index = random.randint(0,len(self.cateid_2_dino_ann[str(neg_cls)])-1)
                    extra_ann = self.cateid_2_dino_ann[str(neg_cls)][extra_ann_index]
                    extra_npy_feat = np.load(os.path.join(self.dino_feats_folder,str(neg_cls),str(extra_ann)))
                    extra_sample = torch.from_numpy(extra_npy_feat)
                    if self.support_norm:
                        extra_sample = extra_sample/torch.linalg.norm(extra_sample)
                    neg_sample_list.append(extra_sample)
                neg_sample = torch.mean(torch.stack(neg_sample_list),dim=0)
            neg_samples.append(neg_sample)
        #negative sample processing done===========================================================================================
        
        #random mapping processing================================================================================================
        num_cls_all = num_cls_neg + len(set(pos_catids))
        if self.region_detect:
            num_cls_all = num_cls_neg + len(tag_bboxes)
        support_samples = torch.zeros(num_cls_all,self.support_feat_dim)
        unique_elements = list(set(pos_catids))
        bos,eos,mem = 0,num_cls_all-1,[]
        random_mapping = {}
        for element in unique_elements:
            rand = random.randint(bos,eos)
            if rand not in mem:
                mem.append(rand)
                random_mapping[element] = rand
            else:
                while rand in mem:
                    rand = random.randint(bos,eos)
                mem.append(rand)
                random_mapping[element] = rand
        tag_labels_list = [random_mapping[element] for element in pos_catids]
        reverse_mapping = {v:k for k,v in random_mapping.items()}
        #random mapping processing done===========================================================================================

        #gathering all the data===================================================================================================
        if self.region_detect:
            tag_labels_list = random.sample(range(1,num_cls_all),len(tag_bboxes))
        tag_labels = torch.LongTensor(tag_labels_list)
        #print(tag_labels)
        tag_bboxes = torch.Tensor(tag_bboxes)
        #unique target index
        tag_indexes = torch.Tensor(list(set(tag_labels_list))).type_as(tag_labels)
        neg_indexes = torch.Tensor(list(set(range(num_cls_all))-set(tag_indexes.tolist()))).type_as(tag_labels)
        #average cls2sample
        if self.mixed_support_selection:
            for key in tag_indexes:
                pos_samples.append(cls2sample[reverse_mapping[int(key)]][0])
        elif self.region_detect:
            pos_samples = pos_region_samples
        else:
            for key in tag_indexes:
                pos_samples.append(torch.mean(torch.stack(cls2sample[reverse_mapping[int(key)]]),dim=0))
        support_samples[tag_indexes] = torch.stack(pos_samples)
        support_samples[neg_indexes] = torch.stack(neg_samples)
        if self.bg_feat is not None:
            support_samples[-1] = self.bg_feat/torch.linalg.norm(self.bg_feat) if self.support_norm else self.bg_feat
        else:
            support_samples[-1] = torch.ones(self.support_feat_dim)/torch.linalg.norm(torch.ones(self.support_feat_dim)) if self.support_norm else torch.ones(self.support_feat_dim) if self.use_bg else support_samples[0]
        cur_iter_pos_text = torch.zeros(num_cls_all,512)
        if self.use_text:
            cur_iter_pos_text[tag_indexes] = torch.stack([self.category_text_features[reverse_mapping[int(key)]][0]/torch.linalg.norm(self.
            category_text_features[reverse_mapping[int(key)]][0]) for key in tag_indexes])
        #cur_iter_pos_text[0] = self.text_background_feat[0]/torch.linalg.norm(self.text_background_feat[0])
        target = {'boxes': tag_bboxes, 'labels': tag_labels}
        if self.return_mask:
            target['masks'] = masks
        if self.aug and self.training:
            query_img,target= self.load_image(query_np,target) if query_np is not None else self.load_image(np.zeros((480,480,3),dtype=np.uint8),target)
            target.pop('size',None)
            target.pop('area',None)
        else:
            query_img,target = self.load_image(query_np,target) if query_np is not None else self.load_image(np.zeros((480,480,3),dtype=np.uint8),target)
        sample = []
        sample.append(query_img)
        sample.append(support_samples.transpose(0,1))
        sample.append(target)
        sample.append(cur_iter_pos_text.transpose(0,1))
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
            sample.append(image_ann)
            return tuple(sample)
def collate_fn(batch):
    batch = list(zip(*batch))
    query_sample, support_sample, targets,text = batch
    query_sample = nested_tensor_from_tensor_list(query_sample)
    support_samples = nested_tensor_from_tensor_list(support_sample)
    text_samples = nested_tensor_from_tensor_list(text)
    return tuple([query_sample, support_samples, targets,text_samples])
def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    ann_path = "./annotations/instances_val2017.json"
    image_folder = "./"
    train = False
    ds = COCODataset(ann_path,image_folder,dino_feats_folder=None,train=train)
    sampler_train = torch.utils.data.RandomSampler(ds)
    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, 8, drop_last=True
    )
    data_loader = DataLoader(
        ds, batch_sampler=batch_sampler_train, collate_fn=collate_fn, num_workers=0
    )
    for sample in data_loader:
        print(len(sample))
    
