"""03. Predict with pre-trained YOLO models
==========================================

This article shows how to play with pre-trained YOLO models with only a few
lines of code.

First let's import some necessary libraries:
"""

from gluoncv import model_zoo, data, utils
from matplotlib import pyplot as plt

######################################################################
# Load a pretrained model
# -------------------------
#
# Let's get an YOLOv3 model trained with on Pascal VOC
# dataset with Darknet53 as the base model. By specifying
# ``pretrained=True``, it will automatically download the model from the model
# zoo if necessary. For more pretrained models, please refer to
# :doc:`../../model_zoo/index`.

# net = model_zoo.get_model('yolo3_darknet53_voc', pretrained=True)
net = model_zoo.get_model('yolo3_mobilenet1.0_voc', pretrained=False)
# net.load_parameters('yolo3_mobilenet1.0_voc_best.params')
net.load_parameters('yolo3_mobilenet1.0_voc_0060_0.0507.params')
######################################################################
# Pre-process an image
# --------------------
#
# Next we download an image, and pre-process with preset data transforms. Here we
# specify that we resize the short edge of the image to 512 px. You can
# feed an arbitrarily sized image.
# Once constraint for YOLO is that input height and width can be divided by 32.
#
# You can provide a list of image file names, such as ``[im_fname1, im_fname2,
# ...]`` to :py:func:`gluoncv.data.transforms.presets.yolo.load_test` if you
# want to load multiple image together.
#
# This function returns two results. The first is a NDArray with shape
# `(batch_size, RGB_channels, height, width)`. It can be fed into the
# model directly. The second one contains the images in numpy format to
# easy to be plotted. Since we only loaded a single image, the first dimension
# of `x` is 1.
MY_CLASSES =  ['Unknown', 'Compacts', 'Sedans', 'SUVs', 'Coupes',
            'Muscle', 'SportsClassics', 'Sports', 'Super', 'Motorcycles',
            'OffRoad', 'Industrial', 'Utility', 'Vans', 'Cycles',
            'Boats', 'Helicopters', 'Planes', 'Service', 'Emergency']
type = 'train'
INPUT_DIR = '/Users/cindyshao/Dropbox/ME599/project/rob535-fall-2019-task-1-image-classification/'
dir = INPUT_DIR + 'data-2019/' + type + '/'
fname = '2983f6f8-00db-4aa3-8ea1-562791e22771/0021_image.jpg'
im_fname = dir + fname
x, img = data.transforms.presets.yolo.load_test(im_fname, short=512)
print('Shape of pre-processed image:', x.shape)


######################################################################
# Inference and display
# ---------------------
#
# The forward function will return all detected bounding boxes, and the
# corresponding predicted class IDs and confidence scores. Their shapes are
# `(batch_size, num_bboxes, 1)`, `(batch_size, num_bboxes, 1)`, and
# `(batch_size, num_bboxes, 4)`, respectively.
#
# We can use :py:func:`gluoncv.utils.viz.plot_bbox` to visualize the
# results. We slice the results for the first image and feed them into `plot_bbox`:

class_IDs, scores, bounding_boxs = net(x)

ax = utils.viz.plot_bbox(img, bounding_boxs[0], scores[0],
                         class_IDs[0], class_names=MY_CLASSES)
plt.show()
