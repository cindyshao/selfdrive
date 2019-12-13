"""03. Predict with pre-trained YOLO models
==========================================

This article shows how to play with pre-trained YOLO models with only a few
lines of code.

First let's import some necessary libraries:
"""

from gluoncv import model_zoo, data, utils
from matplotlib import pyplot as plt
import numpy as np
from glob import glob
import csv
import os
import sys

INPUT_DIR = '/Users/cindyshao/Dropbox/ME599/project/rob535-fall-2019-task-1-image-classification/'
OUTPUT_DIR = '/Users/cindyshao/Dropbox/ME599/'
def main(type):
    ######################################################################
    # Load a pretrained model
    # -------------------------
    #
    # Let's get an YOLOv3 model trained with on Pascal VOC
    # dataset with Darknet53 as the base model. By specifying
    # ``pretrained=True``, it will automatically download the model from the model
    # zoo if necessary. For more pretrained models, please refer to
    # :doc:`../../model_zoo/index`.

    net = model_zoo.get_model('yolo3_darknet53_voc', pretrained=True)
    # net = model_zoo.get_model('yolo3_mobilenet1.0_voc', pretrained=False)
    # net.load_parameters('yolo3_mobilenet1.0_voc_best.params')
    file_name = INPUT_DIR + 'labels_' + type + '.csv'
    test_data = np.loadtxt(file_name, skiprows=1, dtype=str, delimiter=',')
    labels = test_data[:, 1].astype(np.uint8)
    files = test_data[:, 0]
    name = OUTPUT_DIR + '/'+ type + '_labels.csv'
    with open(name, 'w') as f:
        writer = csv.writer(f, delimiter=',', lineterminator='\n')
        writer.writerow(['guid/image', 'label'])
        for i, file in enumerate(files):
            guid = file.split('/')[-2]
            idx = file.split('/')[-1]
            name = file + '_image.jpg'
            im_score = process(name,net,type)
            label = labels[i]
            line = ['{}/{}'.format(guid, idx), label]
            line.extend(im_score)
            writer.writerow(line)


def process(fname,net,type):
    dir = INPUT_DIR + 'data-2019/' + type + '/'
    im_fname = dir + fname
    num_class = 20
    if not os.path.isfile(im_fname):
        return [-5] * num_class

    x, img = data.transforms.presets.yolo.load_test(im_fname, short=512)

    class_IDs, scores, bounding_boxs = net(x)

    class_IDs = class_IDs.asnumpy()
    class_IDs = class_IDs.astype(np.int8)
    scores = scores.asnumpy()
    bounding_boxs = bounding_boxs.asnumpy()
    # im_score = np.zeros(num_class)
    im_score = np.zeros(2*num_class)

    for i, class_ID in enumerate(class_IDs[0, :, 0]):
        if class_ID > 0:
            im_score[2*class_ID] = scores[0][i][0]
            xmin, ymin, xmax, ymax = bounding_boxs[0][i]
            size = (xmax-xmin)*(ymax-ymin)
            im_score[2*class_ID+1] = size

    return im_score


if __name__ == "__main__":
    type = sys.argv[1]
    main(type)

