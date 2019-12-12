from gluoncv import model_zoo, data, utils
import mxnet as mx
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
    ctx = mx.gpu(0)
    net = model_zoo.get_model('mask_rcnn_resnet50_v1b_coco', pretrained=True, ctx = ctx)
    file_name = INPUT_DIR + 'labels_' + type + '.csv'
    test_data = np.loadtxt(file_name, skiprows=1, dtype=str, delimiter=',')
    labels = test_data[:, 1].astype(np.uint8)
    files = test_data[:, 0]
    name = OUTPUT_DIR + '/ '+ type + '_labels.csv'
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
    num_class = 80
    if not os.path.isfile(im_fname):
        return [-5] * num_class

    x, img = data.transforms.presets.rcnn.load_test(im_fname)

    ids, scores, bboxes, masks = [xx[0].asnumpy() for xx in net(x)]

    # class_IDs = class_IDs.asnumpy()
    class_IDs = ids.astype(np.int8)
    # scores = scores.asnumpy()
    im_score = np.zeros(2*num_class)
    for i, class_ID in enumerate(class_IDs):
        if class_ID > 0:
            im_score[2*class_ID] = scores[i]
            xmin, ymin, xmax, ymax = bboxes[i]
            size = (xmax-xmin)*(ymax-ymin)
            im_score[2*class_ID+1] = size

    return im_score


if __name__ == "__main__":
    type = sys.argv[1]
    main(type)

