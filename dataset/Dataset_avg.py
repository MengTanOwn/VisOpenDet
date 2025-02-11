from torch.utils.data import dataset
import copy
from pycocotools.coco import COCO
try:
    import dataset.transforms as T
    from model.utils import nested_tensor_from_tensor_list
except:
    import transforms as T
    from model.utils import nested_tensor_from_tensor_list
from PIL import Image
import os
import cv2
import numpy as np
import random
import torch
import torch.nn.functional as F
class COCODatasetAVG(dataset.Dataset):
    def __init__(self,ann_path,image_folder,dino_feats_folder=None,support_feat_dim=384,max_support_len=81,extra_shots=0,aug=False,strong_aug=False,train=False,support_norm=False,return_mask=False,mixed_support_selection=False,val_cls_ids=None,region_detect=False,use_text=False,bg_feat=None,use_bg=True):
        super(COCODatasetAVG, self).__init__()
        self.bg_feat = torch.load(bg_feat) if bg_feat is not None else None
        self.coco = COCO(ann_path)
        self.image_folder = image_folder
        self.image_ids = self.coco.getImgIds()
        self.cateid_2_dino_feat = {}
        self.dino_feats_folder = dino_feats_folder
        self.extra_shots = extra_shots
        self.max_support_len = max_support_len
        self.support_feat_dim = support_feat_dim
        self.training = train
        self.aug = aug
        self.strong_aug = strong_aug
        self.support_norm = support_norm
        self.return_mask = return_mask
        self.vocclsid_2_cococlsid = {
                    1: 15,
                    2: 2,
                    3: 16,
                    4: 9,
                    5: 44,
                    6: 6,
                    7: 3,
                    8: 17,
                    9: 62,
                    10: 21,
                    11: 67,
                    12: 18,
                    13: 19,
                    14: 4,
                    15: 1,
                    16: 64,
                    17: 20,
                    18: 63,
                    19: 7,
                    20: 72
        }
        self.cococlsid_2_vocclsid = {v:k for k,v in self.vocclsid_2_cococlsid.items()}
        if self.dino_feats_folder:
            for cates in os.listdir(self.dino_feats_folder):
                self.cateid_2_dino_feat[cates] = os.listdir(os.path.join(self.dino_feats_folder,cates))
        else:
            cate = self.coco.getCatIds()
            self.cateid_2_dino_feat = {str(each):[str(each)+".pth"] for each in cate}
    def __len__(self):
        return len(self.image_ids)
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

    def __getitem__(self, index):
        tag_bboxes, pos_catids, pos_samples, neg_samples, memory_category = [], [], [], [],[]
        cls2sample = {}
        try:
            image_ann = self.coco.loadAnns(self.coco.getAnnIds(self.image_ids[index],iscrowd=False))
        except:
            image_ann = self.coco.loadAnns(self.coco.getAnnIds(self.image_ids[index]))
        copy_image_ann = copy.deepcopy(image_ann)
        if image_ann == []: return self.__getitem__(random.randint(0,len(self.image_ids)-1))
        image_info = self.coco.loadImgs(self.image_ids[index])[0]
        #locate the last /
        #image_info['file_name'] = image_info['file_name'].split("/")[-1]
        image_path = os.path.join(self.image_folder,image_info['file_name']) if image_info.get("file_name") is not None else os.path.join(self.image_folder,str(image_info['id']).zfill(12)+".jpg")
        query_np = cv2.imread(image_path)
        #positive sample processing================================================================================================
        for index_ann,each in enumerate(image_ann):
            xyxy = [each['bbox'][0],each['bbox'][1],each['bbox'][0]+each['bbox'][2],each['bbox'][1]+each['bbox'][3]]
            tag_bboxes.append(xyxy)
            if 'voc' in self.image_folder.lower():
                each["category_id"] = self.vocclsid_2_cococlsid[each["category_id"]]
            pos_catids.append(each["category_id"])
            pos_sample = torch.load(os.path.join(self.dino_feats_folder,str(each["category_id"]),"avg_feats.pth")) if self.dino_feats_folder is not None else None
            if self.support_norm:
                pos_sample = pos_sample/torch.linalg.norm(pos_sample)
            if each["category_id"] not in memory_category:
                memory_category.append(each["category_id"])
                cls2sample[each["category_id"]] = pos_sample
                #pos_samples.append(pos_sample[0])
        #positive sample processing done===========================================================================================
        
        #negative sample processing================================================================================================
        num_cls_neg = random.randint(1, self.max_support_len - len(set(pos_catids)))
        if not self.training:
            num_cls_neg = 1
        for _ in range(num_cls_neg):
            neg_cls = random.choice(list(set(self.cateid_2_dino_feat.keys())-set(pos_catids)))
            neg_sample = torch.load(os.path.join(self.dino_feats_folder,str(neg_cls),"avg_feats.pth")) if self.dino_feats_folder is not None else None
            if self.support_norm:
                neg_sample = neg_sample/torch.linalg.norm(neg_sample)
            neg_samples.append(neg_sample)
        #negative sample processing done===========================================================================================
        
        #random mapping processing================================================================================================
        num_cls_all = num_cls_neg + len(set(pos_catids))
        support_samples = torch.zeros(num_cls_all,self.support_feat_dim)
        unique_elements = list(set(pos_catids))
        BOS,EOS,mem = 0,num_cls_all-1,[]
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
        reverse_mapping = {v:k for k,v in random_mapping.items()}
        #random mapping processing done===========================================================================================

        #gathering all the data===================================================================================================
        tag_labels = torch.LongTensor(tag_labels_list)
        tag_bboxes = torch.Tensor(tag_bboxes)
        #unique target index
        tag_indexes = torch.Tensor(list(set(tag_labels_list))).type_as(tag_labels)
        neg_indexes = torch.Tensor(list(set(range(num_cls_all))-set(tag_indexes.tolist()))).type_as(tag_labels)
        for key in tag_indexes:
            pos_samples.append(cls2sample[reverse_mapping[int(key)]])
        support_samples[tag_indexes] = torch.stack(pos_samples)
        support_samples[neg_indexes] = torch.stack(neg_samples)


        if self.support_norm:
            support_samples[-1] = torch.ones(self.support_feat_dim)/torch.linalg.norm(torch.ones(self.support_feat_dim))
        else:
            support_samples[-1] = torch.ones(self.support_feat_dim)
        if not self.training:
            support_list = []
            iter_dir = os.listdir(self.dino_feats_folder)
            #sort from low to high
            iter_dir.sort(key=lambda x:int(x))
            for cls_coco in iter_dir:
                try:
                    feat = os.path.join(self.dino_feats_folder,str(cls_coco),'avg_feats.pth')
                    feat = torch.load(feat)
                except:
                    continue
                feat = feat/torch.linalg.norm(feat) if self.support_norm else feat
                support_list.append(feat)
            if self.bg_feat is not None:
                support_list += [self.bg_feat/torch.linalg.norm(self.bg_feat)]
            else:
                support_list += [torch.ones(self.support_feat_dim)/torch.linalg.norm(torch.ones(self.support_feat_dim))]
            support_samples = torch.stack(support_list)
            # cates_for_test = os.listdir(self.dino_feats_folder)
            # cates_for_test.sort(key=lambda x:int(x))
            reverse_mapping = {k:v for k,v in enumerate(iter_dir)}
            if 'voc' in self.image_folder.lower():
                reverse_mapping = {k:self.cococlsid_2_vocclsid[v] for k,v in reverse_mapping.items()}
        target = {'boxes': tag_bboxes, 'labels': tag_labels}
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
def collate_fn_avg(batch):
    batch = list(zip(*batch))
    try:
        query_sample, support_sample, targets = batch
        query_sample = nested_tensor_from_tensor_list(query_sample)
        support_samples = nested_tensor_from_tensor_list(support_sample)
        return tuple([query_sample, support_samples, targets,None])#NOTE None is for CLIP training
    except:
        query_sample, support_sample, targets, hw, reverse_mapping, imgid, ann = batch
        query_sample = nested_tensor_from_tensor_list(query_sample)
        support_samples = nested_tensor_from_tensor_list(support_sample)
        return tuple([query_sample, support_samples, targets, hw, reverse_mapping, imgid, ann])
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    ann_path = "./annotations/instances_val2017.json"
    image_folder = "./"
    train = True
    ds = COCODatasetAVG(ann_path,image_folder,dino_feats_folder=None,train=train)
    sampler_train = torch.utils.data.RandomSampler(ds)
    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, 8, drop_last=True
    )
    data_loader = DataLoader(
        ds, batch_sampler=batch_sampler_train, collate_fn=collate_fn_avg, num_workers=0
    )
    for sample in data_loader:
        print(len(sample))
    
