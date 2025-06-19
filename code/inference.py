import json
from mmdet.apis import init_detector, inference_detector_kaist,show_result_pyplot
import os
from pycocotools.coco import COCO
from tqdm import tqdm
import cv2
import numpy as np
dataset = 'LLVIP'
if dataset =='FLIR' or dataset =='FLIR_test':
    categories_map = {"person": 0, "car": 1, "bicycle": 2}
elif dataset == 'LLVIP' or dataset == 'LLVIP_test':
    categories_map = {"person": 0}
elif dataset =='M3FD' or dataset =='M3FD_test':
    categories_map = {"car": 1, "Truck": 2, "People":3, "Bus":4, "Lamp":5, "Motor.":6}


def make_test_json(input_dir):
    data_dir = input_dir
    image_file_dir = os.path.join(data_dir, 'rgb')
    annotations_info = {'images': [], 'annotations': [], 'categories': []}

    for key in categories_map:
        categoriy_info = {"id":categories_map[key], "name":key}
        annotations_info['categories'].append(categoriy_info)

    file_names = sorted([image_file_name.split('.')[0]
                for image_file_name in os.listdir(image_file_dir)])
    for i, file_name in enumerate(file_names):
        # print(i)
        image_file_name = file_name + '.jpg'

        image_file_path = os.path.join(image_file_dir, image_file_name)
        image_info = dict()
        image = cv2.cvtColor(cv2.imread(image_file_path), cv2.COLOR_BGR2RGB)
        height, width, _ = image.shape
        image_info = {'file_name': image_file_name, 'id': i+1,
                    'height': height, 'width': width}
        annotations_info['images'].append(image_info)
    end_path = input_dir +'/test.json'
    print('============',end_path)
    with  open(end_path, 'w')  as f:
        json.dump(annotations_info, f, indent=4)

# root = '/home/shen2/zhb/doublecodetrIDAT/'+dataset+'/rgb/'
# json_root = '/home/shen2/zhb/doublecodetrIDAT'+dataset
root='/home/shen5/zhb/ODinMJ/test/rgb'
# json_root ='/home/shen5/zhb/ODinMJ/test'
# make_test_json(json_root)
# anno_path = 'test.json'
# coco = COCO(json_root +'/test.json')
# ids = list(coco.imgs.keys())

filename_id = {}

# 获取文件名与对应的 ID
# for idx in tqdm(range(len(ids))):
#     img_id = ids[idx]
#     img_ids = coco.getImgIds(imgIds=img_id)
#     image = coco.loadImgs(img_ids)
#     for i in image:
#         filename_id[i['file_name']] = i['id']

# 目标检测配置文件
loop =4
config_file = '/home/shen2/zhb/doublecodetrIDAT/code/configs/codino_vit_twostream_640_autoaugv1_train1.py'
# checkpoint_file = 'D:/master/double-co-detr/weight/codino_vit_twostream_640_autoaugv1_train1_IDAT_loop'+str(loop)+'/best_IDAT_FLIR_'+str(loop)+'.pth'
checkpoint_file = '/home/shen5/zhb/2025_5_9/codino_vit_twostream_640_autoaugv1_train1_IDAT_ODinMJ/best_bbox_mAP_epoch_6.pth'
# 配置模型
model = init_detector(config=config_file,
                      checkpoint=checkpoint_file,
                      device='cuda:0')

# 需要推理的图片的路径

# output_dir = '/home/shen2/zhb/doublecodetrIDAT/'+dataset+'/result/'  # 保存结果的路径
output_dir = '/home/shen5/zhb/ODinMJ/test/result/idat/'
os.makedirs(output_dir, exist_ok=True)

# 存储结果，并生成 json
results = []

# 开始推理
for file in tqdm(os.listdir(root)):
    img_path = os.path.join(root, file)
    text_filename = os.path.join(output_dir,(os.path.splitext(file)[0]+'.txt'))

    result = inference_detector_kaist(model=model, img=img_path)
    img = cv2.imread(img_path)
    result_yolo=[]
    for cate, items in enumerate(result[0]):
        # 同一类别的有很多结果
        for item in items:
            # item = item.tolist()
            x, y, w, h, s = item[0], item[1], item[2], item[3], item[4]
            # xmin=x-w/2
            # ymin=y-h/2
            # xmax=x+w/2
            # ymax=y+h/2
            # result_yolo.append([cate,s,xmin,ymin,xmax,ymax])
            result_yolo.append([cate,s,x,y,w,h])
            # d = {}
            # # d['image_id'] = filename_id[file]
            # d['category_id'] = cate
            # d['bbox'] = [x, y, w, h]
            # d['score'] = s
            # results.append(d)
    with open(text_filename,'w') as f:
        for res in result_yolo:
            line = ' '.join(map(str,res)) + '\n'
            f.write(line)
    # img = show_result_pyplot(model, img_path, result, score_thr=0.3)
        # os.makedirs(output_dir+file.split('.')[0]+'/', exist_ok=True)
        # cv2.imwrite("{}/{}".format(output_dir+file.split('.')[0]+'/NiN/', ''+file), img)
    # cv2.imwrite("{}/{}".format(output_dir+'/NiN/', file), img)
# 保存推理结果为 JSON
# with open(output_dir+'result.json', 'w') as f:
#     json.dump(results, f, indent=4)
