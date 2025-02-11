import torch
import os
import cv2
def square_resize(img, dsize,value=(0,0,0)):
    """
    按照图片的长边扩充为方形, 再resize到指定大小
    """
    h, w, _ = img.shape
    top, bottom, left, right = 0, 0, 0, 0
    if h > w:
        diff = h - w
        left = int(diff / 2)
        right = diff - left
    else:
        diff = w - h
        top = int(diff / 2)
        bottom = diff - top
    img_new = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=value)
    return cv2.resize(img_new, dsize)
def support_set_preprocess(classes,support_set_path):
    try:
        support_set = {}
        for cls in classes[1:]:
            if not os.path.exists(os.path.join(support_set_path, cls)):
                break
            support_set[cls] = []
            cur_cls_imgs = os.listdir(os.path.join(support_set_path, cls))
            for img_path in cur_cls_imgs:
                if img_path.endswith(".png") or img_path.endswith(".jpg"):
                    support_set[cls].append(os.path.join(support_set_path, cls, img_path))
            if len(support_set[cls]) == 0:
                support_set.pop(cls)
        return support_set
    except:
        return None
def load_weight(args,model_without_ddp):
    checkpoint = torch.load(args.resume)#, map_location="cpu")
    try:
        pretrained_dict = checkpoint.get("model") if checkpoint.get("model") is not None else checkpoint
        model_without_ddp.load_state_dict(pretrained_dict)
    except Exception as e:
        print(e)
        pretrained_dict = checkpoint
        model_dict = model_without_ddp.state_dict()
        # match layer state and shape
        if args.use_mask_head:
            if 'tide' in list(model_dict.keys())[0]:
                pretrained_dict = {
                    'tide.'+k: v
                    for k, v in pretrained_dict.items()
                    if 'tide.'+k in model_dict and v.shape == model_dict['tide.'+k].shape
                }
            else:
                pretrained_dict = {
                k: v
                for k, v in pretrained_dict.items()
                if k in model_dict and v.shape == model_dict[k].shape
        }
        else:
            pretrained_dict = {
                k: v
                for k, v in pretrained_dict.items()
                if k in model_dict and v.shape == model_dict[k].shape
            }
        model_dict.update(pretrained_dict)
        model_without_ddp.load_state_dict(model_dict)
    return model_without_ddp


