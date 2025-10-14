# Copyright (2023) Bytedance Ltd. and/or its affiliates 


import os
from datasets.coco_eval import CocoEvaluator as DefaultCocoEvaluator
import logging
import numpy as np


class CocoEvaluator(DefaultCocoEvaluator):
    def synchronize_between_processes(self):
        for iou_type in self.iou_types:
            if len(self.eval_imgs[iou_type]) == 0:
                self.eval_imgs[iou_type] = [np.zeros((1, 4, 0))]
        super().synchronize_between_processes()

    def summarize(self, save_map):
        super().summarize()
        stats = self.coco_eval['bbox'].stats * 100.0
        logger = logging.getLogger(__name__)
        os.makedirs(save_map, exist_ok=True)  # 경로가 없으면 생성
        log_file = os.path.join(save_map, "mAP_info.txt")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            file_handler = logging.FileHandler(log_file, mode='w')
            file_handler.setLevel(logging.INFO)

            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)

            logger.addHandler(file_handler)
        logger.info("\n\
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = {:.3f}\n\
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = {:.3f}\n\
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = {:.3f}\n\
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = {:.3f}\n\
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = {:.3f}\n\
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = {:.3f}\n\
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = {:.3f}\n\
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = {:.3f}\n\
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = {:.3f}\n\
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = {:.3f}\n\
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = {:.3f}\n\
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = {:.3f}\n".format(*(stats.tolist())))

