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


def main():
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
    test_data = np.loadtxt('/Users/cindyshao/Dropbox/ME599/project/rob535-fall-2019-task-1-image-classification/labels.csv', skiprows=1, dtype=str, delimiter=',')
    labels = test_data[:, 1].astype(np.uint8)
    files = test_data[:, 0]
    name = '{}/trainval_labels.csv'.format('/Users/cindyshao/Dropbox/ME599/project/')
    with open(name, 'w') as f:
        writer = csv.writer(f, delimiter=',', lineterminator='\n')
        writer.writerow(['guid/image', 'label'])
        for i, file in enumerate(files):
            guid = file.split('/')[-2]
            idx = file.split('/')[-1]
            name = file + '_image.jpg'
            im_score = process(name,net)
            label = labels[i]
            line = ['{}/{}'.format(guid, idx), label]
            line.extend(im_score)
            writer.writerow(line)


def process(fname,net):
    # im_fname = utils.download('https://raw.githubusercontent.com/zhreshold/' +
    #                           'mxnet-ssd/master/data/demo/dog.jpg',
    #                           path='dog.jpg')
    # im_fname = '/Users/cindyshao/Dropbox/ME599/project/rob535-fall-2019-task-1-image-classification/data-2019/trainval/01aaa345-52ad-4939-8207-2d39c11acfdc/0013_image.jpg'
    # im_fname = '/Users/cindyshao/Dropbox/ME599/project/rob535-fall-2019-task-1-image-classification/data-2019/trainval/ff87e400-2d5d-4e95-a840-a58c538d6ef0/0000_image.jpg'
    dir = '/Users/cindyshao/Dropbox/ME599/project/rob535-fall-2019-task-1-image-classification/data-2019/trainval/'
    im_fname = dir + fname
    if not os.path.isfile(im_fname):
        return [-5] * 100

    x, img = data.transforms.presets.yolo.load_test(im_fname, short=512)
    # print('Shape of pre-processed image:', x.shape)

    ######################################################################

    class_IDs, scores, bounding_boxs = net(x)
    # ax = utils.viz.plot_bbox(img, bounding_boxs[0], scores[0],
    #                          class_IDs[0], class_names=net.classes)
    # plt.show()

    class_IDs = class_IDs.asnumpy()
    class_IDs = class_IDs.astype(np.int8)
    scores = scores.asnumpy()
    im_score = np.zeros(100)

    for i, class_ID in enumerate(class_IDs[0, :, 0]):
        if class_ID > 0:
            im_score[class_ID] = scores[0][i][0]

    return im_score


if __name__ == "__main__":
    main()

