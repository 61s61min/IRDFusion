import json

from mmdet.core.visualization import imshow_det_bboxes

# from code.mmdet.core.visualization import imshow_det_bboxes
from mmdet.apis import init_detector, inference_detector_kaist,show_result_pyplot
import os
from pycocotools.coco import COCO
from tqdm import tqdm
import cv2
import numpy as np
dataset = 'FLIR_test'





root = 'D:/master/double-co-detr/'+dataset+'/rgb/'
json_root = 'D:/master/double-co-detr/'+dataset



output_dir = 'D:/master/double-co-detr/'+dataset+'/result/'  # 保存结果的路径
os.makedirs(output_dir, exist_ok=True)



# 开始推理
for file in tqdm(os.listdir(root)):
    img_path = os.path.join(root, file)
    img = cv2.imread(img_path)
    bbox = []
    label = []
    txt_file = json_root+'/result/labels/'+file.split('.')[0]+'_PreviewData.txt'
    h,w = img.shape[:2]
    with open(txt_file, 'r') as labels:
        for line in labels:
            values = line.strip().split()
            label.append(int(values[0]))
            bbox.append([float(values[1])*w-float(values[3])*w/2,float(values[2])*h-float(values[4])*h/2,float(values[1])*w+float(values[3])*w/2,float(values[2])*h+float(values[4])*h/2,1.0]) # 使用strip()去除每行的换行符



    img = imshow_det_bboxes(
            img,
            np.array(bbox),
            np.array(label),
            # class_names=('People', 'Car', 'Bus', 'Lamp', 'Motor.', 'Truck'),
            class_names=('person','car','bicycle'),
            score_thr=0,
            bbox_color='red',
            text_color='red',
            mask_color=None,
            thickness=2,
            font_size=8,
            win_name='result',
            show=False,
            wait_time='',
            out_file=None)
    cv2.imwrite("{}/{}".format(output_dir + '/groundtruth/', file), img)
    # cv2.imwrite("{}/{}".format(output_dir + '/groundtruth/', file.split('.')[0]+'IDAT.'+file.split('.')[1]), img)