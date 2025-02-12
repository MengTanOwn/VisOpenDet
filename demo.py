import os
import cv2
import tkinter as tk
from tkinter import simpledialog

from model.inference import VisInference

import argparse

parser = argparse.ArgumentParser(description='VisOpenDet Demo')
parser.add_argument('--image_path', type=str, default='test_images/3.jpg')
parser.add_argument('--pre_score', type=float, default=0.4)
args = parser.parse_args()

model=VisInference()

def random_color():
    import random
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

img_size = 1024

#inference image
img_path = args.image_path
cv2_img = cv2.imread(img_path)


copy_org_img = cv2_img.copy()
copy2_org_img = cv2_img.copy()

show_result_img = cv2_img.copy()


coord = []
#draw box with mouse
def draw_box(event, x, y, flags, param):
    global x0, y0, drawing, mode, copy_org_img,coord
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        x0, y0 = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            copy_org_img = copy2_org_img.copy()
            cv2.rectangle(copy_org_img, (x0, y0), (x, y), (0, 255, 0), 2)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.rectangle(copy_org_img, (x0, y0), (x, y), (0, 255, 0), 2)
        coord.append([x0, y0, x, y])
        print(coord)

cv2.namedWindow('image',0)
cv2.namedWindow('query_image',0)
cv2.setMouseCallback('image', draw_box)

drawing = False
x0, y0 = -1, -1
prompt_coord = []
color_map = {}
for cls_id in list(range(0,26)):
    color_map[cls_id] = random_color()
cates = []
continue_ = True
skip_label = False
POS_THRES_ = args.pre_score
while True:
    if continue_:
        cv2.imshow('image', copy_org_img)
    #press "s" to save the box and continue
    keycode = cv2.waitKey(5)
    if keycode == ord('s'):
        prompt_coord.append(coord[-1])
        color = random_color()
        print('prompt_coord:',prompt_coord)
        root = tk.Tk()
        #隐藏tk窗口
        root.withdraw()
        entry_int = simpledialog.askinteger("Support Set", "Enter the integer class of the object:")
        #move the dialog window to the center of the screen
        root.update_idletasks()
        root.geometry(f"+{int((root.winfo_screenwidth() - root.winfo_reqwidth()) / 2)}+{int((root.winfo_screenheight() - root.winfo_reqheight()) / 2)}")

        cates.append(entry_int)
        #destroy tkinter window
        root.destroy()
        if color_map.get(entry_int) is None:
            color_map[entry_int] = color
        cv2.rectangle(copy2_org_img, (coord[-1][0], coord[-1][1]), (coord[-1][2], coord[-1][3]), color_map[entry_int], 2)
        cv2.putText(copy2_org_img, str(entry_int), (coord[-1][0], coord[-1][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1,color_map[entry_int],2)
        copy_org_img = copy2_org_img.copy()
        continue
    #press "q" to quit the box prompts and do inference
    elif keycode == ord('q'):
        org_img_to_show = copy_org_img.copy()
        cv2.destroyAllWindows()
        continue_ = False
        prompts = {}
        prompts['prompts'] = []
        for cate_i,box_i in zip(cates,prompt_coord):
            dict_pro = {"category_id": cate_i,
                        "rects": [box_i]}
            prompts['prompts'].append(dict_pro)
        prompts['prompt_image'] = cv2_img
        tag_bboxes = []
        predict = model.interactve_inference(prompts,score_t=POS_THRES_)
        count = 0
        if len(predict["labels"])>0:
            color_map = {}
            for cls_id in predict["labels"]:
                color_map[cls_id] = random_color()
            
            for box,score,cls in zip(predict["boxes"],predict["scores"],predict["labels"]):
                if cls==None:
                    continue
                count += 1
                cv2.rectangle(show_result_img, (box[0], box[1]), (box[2], box[3]), color_map[cls], 2)
                if not skip_label:
                    cv2.putText(show_result_img, str(cls)+':'+str(score)[:4], (box[0], box[1]+10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3,color_map[cls],1)
        
            print('target number is',count)
            cv2.namedWindow('org_image',0)
            cv2.imshow('org_image', org_img_to_show)
            
            cv2.namedWindow('result',0)
            cv2.imshow("result", show_result_img)
            cv2.waitKey(0)
    elif keycode == ord('c'):
        continue_ = True
        cv2.destroyAllWindows()
        copy_org_img = cv2_img.copy()
        cv2_query_img = cv2_img.copy()
        cv2.namedWindow('image')
        cv2.setMouseCallback('image', draw_box)
    elif keycode == ord('f'):
        #adjust POS THRES
        root = tk.Tk()
        root.withdraw()
        root.geometry(f"+{int((root.winfo_screenwidth() - root.winfo_reqwidth()) / 2)}+{int((root.winfo_screenheight() - root.winfo_reqheight()) / 2)}")
        entry_float = simpledialog.askfloat("Support Set", "Enter the integer class of the object:")
        POS_THRES_ = entry_float
        if entry_float == None:
            POS_THRES_ = 0.105
        root.destroy()
    elif keycode == ord('r'):
        prompt_coord = []
        coord = []
        cates = []
        cv2.destroyAllWindows()
        copy_org_img = cv2_img.copy()
        copy2_org_img = cv2_img.copy()
        show_result_img = cv2_img.copy()
        cv2.namedWindow('image')
        cv2.setMouseCallback('image', draw_box)
    elif keycode == ord('e'):
        break