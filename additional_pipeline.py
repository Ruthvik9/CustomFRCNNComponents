import cv2
import time
import random
import torch
import numpy as np
from mmcv.transforms import BaseTransform
from mmdet.structures.bbox.horizontal_boxes import HorizontalBoxes
from mmdet.registry import TRANSFORMS
import albumentations as A

@TRANSFORMS.register_module()
class MyAlbuTransform(BaseTransform):
    def __init__(self, prob=0.8, debug=False):
        self.prob = prob
        self.debug = debug
        self.transforms = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=self.prob),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=self.prob),
            A.RandomRotate90(p=1.0)
        ],bbox_params=A.BboxParams(format='pascal_voc', label_fields=['gt_bboxes_labels']))
        
    def transform(self, results):# results is the dictionary containing the data pertaining to an image.
        bboxes = results['gt_bboxes'].tensor
        # Using the img_shape key to get the width and height because the width and height from the annotation
        # files are incorrect in some cases!!
        width = results['img_shape'][1]
        height = results['img_shape'][0]
        image = results['img']
        if image.dtype not in [np.uint8]:
            raise Exception("Hold up!! image type is different from uint8")
        bboxes_np = bboxes.numpy().tolist() # So now the original bboxes are still in bboxes.
        bboxes_np_debug = np.copy(bboxes_np)
        # Since some of the bounding box coordinates are lesser than 0 (and maybe some of them are greater than the width/height), dealing with them -
        for i,box in enumerate(bboxes_np):
            if box[0] < 0: # xmin
                bboxes_np[i][0] = 0
            if box[1] < 0: # ymin
                bboxes_np[i][1] = 0
            if box[2] > width: # xmax
                bboxes_np[i][2] = width
            if box[3] > height: # ymax
                bboxes_np[i][3] = height
        
        try:
            augmented = self.transforms(image=image, bboxes=bboxes_np, gt_bboxes_labels=results['gt_bboxes_labels'])
        except:
            print("Error encountered in image ",results['img_path'])
            print("Keys values in the results dict so far ",results.items())
            print("Bboxes np ",bboxes_np)
            print("Height and width ",results['img_shape'])
            print("BBoxes old ",bboxes_np_debug)
            raise Exception("STOP! There is some issue with the above bounding boxes.")
        # print("Keys of the augmented dictionary ",augmented.keys())
        results['img'] = augmented['image'].astype(np.uint8)
        results['img_shape'] = results['img'].shape # Updating the img_shape metadata too cuz it's necessary for downstream tasks.
        results['height'] = results['img_shape'][0]
        results['width'] = results['img_shape'][1]
        results['gt_bboxes'] = HorizontalBoxes(torch.tensor([[box[0], box[1], box[2], box[3]] for box in augmented['bboxes']]))
        
        
        return results
