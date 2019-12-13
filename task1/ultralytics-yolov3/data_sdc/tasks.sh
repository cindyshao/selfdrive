#!/bin/bash


conda activate yolov3


## data processing



## train
# 3 labels
python train.py --data data_sdc/sdc.data --cfg cfg/yolov3-sdc.cfg --weights weights/yolov3.weights --name sdc_labels --epochs 160 
# coco 80 labels
python train.py --data data_sdc/sdc_coco.data --cfg cfg/yolov3.cfg --weights weights/yolov3.weights --name coco_labels --epochs 160 

## test

# use sdc-3-labels trained weights

# use sdc-coco-labels trained weights
python detect.py --cfg cfg/yolov3.cfg --weight weights/sdc_best.pt --source /media/ml/Backup/ROB535_Perception/rob535-fall-2019-task-1-image-classification/images/trainval/01aaa345-52ad-4939-8207-2d39c11acfdc
# use yolov3 weights
python detect.py --cfg cfg/yolov3.cfg --weight weights/yolov3.weights --source /media/ml/Backup/ROB535_Perception/rob535-fall-2019-task-1-image-classification/images/trainval/01aaa345-52ad-4939-8207-2d39c11acfdc

