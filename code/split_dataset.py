import json
import shutil

from tqdm import tqdm

json_path ='/home/shen5/zhb/ODinMJ/coco-labels/val.json'
image_rootpath='/home/shen5/zhb/ODinMJ/'
src_rgb_path = '/home/shen5/zhb/ODinMJ/rgb/'
src_tir_path = '/home/shen5/zhb/ODinMJ/tir/'
dst_rgb_path = image_rootpath+'yolo/visible/test/'
dst_tir_path = image_rootpath +'yolo/infrared/test/'
# src_label_path='/home/shen5/zhb/ODinMJ/yolo-labels/'
# dst_label_path = '/home/shen5/zhb/ODinMJ/yolo/labels/train/'
with open(json_path,'r') as f:
    data=json.load(f)
image_name=[image['file_name'] for image in data['images']]
for img in tqdm(image_name,desc='process:'):
    img_id = img.split('.')[0]
    src_rgb_img_path = src_rgb_path+img
    src_tir_img_path = src_tir_path + img
    dst_rgb_img_path = dst_rgb_path+img
    dst_tir_img_path = dst_tir_path + img
    shutil.copy2(src_rgb_img_path,dst_rgb_img_path)
    shutil.copy2(src_tir_img_path, dst_tir_img_path)
    # src_txt_path = src_label_path + img_id+'.txt'
    # dst_txt_path = dst_label_path + img_id+'.txt'
    # shutil.copy2(src_txt_path, dst_txt_path)

