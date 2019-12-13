# Code for task 1

We used [darknet](https://pjreddie.com/darknet/yolo/) for training Yolov3 on our dataset. 

# Data processing
Data files and sripts are placed in folder `data/*`. Please refer to "Training YOLO on VOC" section of [darknet](https://pjreddie.com/darknet/yolo/) for generating files required for training.

Explanation of files:
- `data/car.data`: dataset specifications (required by darknet)
- `data/car.names`: specify names for each label (required by darknet)
- `data/projection_matrix_playground.ipynb`: jupyter notebook to experiment with bounding box extraction
- `data/get_bbox_label.py`: extract projected bounding box coordinates in 2D image and required label files required by darknet.
- `data/get_train_test_files.py`: generate training and testing files specification required by darknet.


# Model Configuration
Model configurations are listed in cfg/*.cfg
- yolov3-car.cfg: configuration used for initial training without negative sampels
- yolov3-car-finetune.cfg: configuration for fine tuning with negative samples
- yolov3-car-detect.cfg: configuration for detecting and inference using trained weights

# Trained and Pre-trained Weights
You can find and download pretrained and trained weights [here](https://drive.google.com/drive/folders/12MWXPy8H_bGNLPH8fNR9CTtFbPmuBemU?usp=sharing).
- `darknet53.conv.74`: pretrained weights given by darknet
- `yolov3-car.weights`: trained weights without negative samples
- `yolov3-car-finetuned.weights`: trained weights with negative samples

# Training 
First, we train without negative samples.
```
./darknet detector train data/car.data cfg/yolov3-car.cfg darknet53.conv.74 > logs/train.log
```

Then we fine tune model with negative samples.
```
./darknet detector train data/car.data cfg/yolov3-car-finetune.cfg yolov3-car.weights > logs/train_finetune.log
```

Useful command to check training loss from training logs.
```
grep "avg" logs/train.log
```

