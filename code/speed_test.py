
import json
import time

from thop import profile


from mmcv.parallel import collate, scatter

from mmdet.apis import init_detector, show_result_pyplot
import os
from pycocotools.coco import COCO
from tqdm import tqdm
import cv2
import numpy as np

from mmdet.apis import inference_detector_kaist
from mmdet.datasets.pipelines import Compose
from torchsummary import summary
dataset = 'FLIR'

root = '/home/shen2/zhb/doublecodetrIDAT/'+dataset+'/rgb/'
json_root = 'D:/master/double-co-detr/'+dataset

config_file = '/home/shen5/zhb/2025_5_9/codino_vit_twostream_640_autoaugv1_train1_qcy_seadronesea/codino_vit_twostream_640_autoaugv1_train1.py'
# checkpoint_file = 'D:/master/double-co-detr/weight/codino_vit_twostream_640_autoaugv1_train1_IDAT_loop'+str(loop)+'/best_IDAT_FLIR_'+str(loop)+'.pth'
# checkpoint_file = '/home/shen2/zhb/2025_1_15/codino_vit_twostream_640_autoaugv1_train1_ISDF_FDOMdualfeedback_FLIR/best_bbox_mAP_50_epoch_12.pth'
checkpoint_file = '/home/shen5/zhb/2025_5_9/codino_vit_twostream_640_autoaugv1_train1_qcy_seadronesea/best_bbox_mAP_50_epoch_26.pth'
# 配置模型
model = init_detector(config=config_file,
                      checkpoint=checkpoint_file,
                      device='cpu')
datas = []
cfg = model.cfg
img = '/home/shen2/zhb/doublecodetrIDAT/FLIR/rgb/FLIR_08864.jpg'
data = dict(img_info=dict(filename=img), img_prefix=None)
# build the data pipeline
for i, pipeline in enumerate(cfg.data.test.pipeline):
    if pipeline['type'] == 'MultiScaleFlipAug':
        assert 'transforms' in pipeline
        for i, transform in enumerate(pipeline['transforms']):
            # pipeline['transforms'] = replace_ImageToTensor(pipeline['transforms'])
            if transform['type'] == 'ImageToTensor':
                pipeline['transforms'][i] = {'type': 'PairedImagesDefaultFormatBundle'}
test_pipeline = Compose(cfg.data.test.pipeline)
data = test_pipeline(data)
# import pdb; pdb.set_trace()
datas.append(data)
device = next(model.parameters()).device
data = collate(datas, samples_per_gpu=1)
# just get the actual data from DataContainer
data['img_metas'] = [img_metas.data[0] for img_metas in data['img_metas']]
data['img'] = [img.data[0] for img in data['img']]
data['img_lwir'] = [img.data[0] for img in data['img_lwir']]
if next(model.parameters()).is_cuda:
    # scatter to specified GPU
    # import pdb; pdb.set_trace()
    data = scatter(data, [device])[0]
flop, params = profile(model, inputs = ( data['img'],data['img_lwir'],data['img_metas'],False))

print('FLOPs = ' +str(flop/1000**3) + 'G')
print('params = '+str(params/1000**2) + 'M')
summary(model, (3,640,640))




# 开始推理
# all_time = 0
# for file in tqdm(os.listdir(root)):
#     img_path = os.path.join(root, file)
#
#     result, start_time,end_time  = inference_detector_kaist(model=model, img=img_path)
#     all_time += end_time-start_time
# print('time per image:')
# print(all_time/len(os.listdir(root))*1000)

