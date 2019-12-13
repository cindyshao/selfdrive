"""Script to generate file with label and bounding box."""
import os
import csv
import numpy as np
from glob import glob
import matplotlib.image as mpimg
from tqdm import tqdm

DATASET_ROOTDIR = '/media/ml/Backup/ROB535_Perception/rob535-fall-2019-task-1-image-classification/'

TRAINVAL_PATH = os.path.join(DATASET_ROOTDIR, 'images/trainval/')

CLASSES = np.loadtxt(os.path.join(DATASET_ROOTDIR, 'classes.csv'), skiprows=1, dtype=str, delimiter=',')
LABELS = CLASSES[:, 2].astype(np.uint8)

# Keep track of image samples with valid label
Labeled_files = []
Labeled_files_path = os.path.join(DATASET_ROOTDIR, 'labeled_trainval_files.txt')

# COCO_LABEL = [2, 7, 1]  # car, truck, cycle
COCO_LABEL = [0, 1, 2]  # car, truck, cycle

# t = [t_x, t_y, t_z]
def mtx_translate(t):
    T_t = np.array([[1, 0, 0, t[0]],[0, 1, 0, t[1]],[0, 0, 1, t[2]],[0, 0, 0, 1]])
    return T_t

# n = [n_x, n_y, n_z]
def mtx_rotation(n):
    # a is anchor point
    # a = np.array([0, 0, 0])
    theta = np.linalg.norm(n)
    T_r = 0
    if theta == 0:
        T_r = np.eye(4)
    else:
        k = n/theta
        K =np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        R = np.eye(3)+np.sin(theta)*K + (1-np.cos(theta))*np.matmul(K,K)
        T_r = np.column_stack((R, np.array([0,0,0]))) 
        T_r = np.vstack((T_r, np.array([0,0,0,1]))) 
        # T_r = mtx_tsranslate(a)*T_r*mtx_translate(-a)
    return T_r

# s = [s_x, s_y, s_z]
def mtx_scale(s):
    # a is anchor point
    # a = np.array([0, 0, 0])
    T_s = np.eye(4)
    T_s[0][0] = s[0]
    T_s[1][1] = s[1]
    T_s[2][2] = s[2]
    # T_s = mtx_tsranslate(a)*T_s*mtx_translate(-a)
    return T_s

def min_bounding_box(coords, crop=True, image_size=(1052, 1914), normalize=True):
    """minimum bounding box for 2xn of coordinates"""
    xmin = np.min(coords[0, :])
    xmax = np.max(coords[0, :])
    ymin = np.min(coords[1, :])
    ymax = np.max(coords[1, :])

    if crop:
        xmin = np.max([0., xmin])
        xmax = np.min([xmax, image_size[1]])
        ymin = np.max([0., ymin])
        ymax = np.min([ymax, image_size[0]])
    
    bbox = np.array([(xmax+xmin)/2, (ymax+ymin)/2, xmax - xmin, ymax - ymin])
    if normalize:
        return np.array([
            bbox[0] / image_size[1],
            bbox[1] / image_size[0],
            bbox[2] / image_size[1],
            bbox[3] / image_size[0]
        ])
    else:
        return bbox


def get_bounding_boxes(bboxs, proj, crop=True, image_size=(1052, 1914), skip_eval=True, normalize=True):
    bounding_boxes = []
    labels = []
    Pcs = []
    for bbox in bboxs:
        if skip_eval and bool(bbox[-1]):
            continue
        R = mtx_rotation(bbox[0:3])
        T = mtx_translate(bbox[3:6])
        a, b, c = bbox[6:9]
        Pb = np.array(
            [
                [a/2, b/2, c/2, 1],
                [a/2, b/2, -c/2, 1],
                [a/2, -b/2, c/2, 1], 
                [a/2, -b/2, -c/2, 1], 
                [-a/2, b/2, c/2, 1], 
                [-a/2, b/2, -c/2, 1], 
                [-a/2, -b/2, c/2, 1], 
                [-a/2, -b/2, -c/2, 1]
            ], dtype='float'
        ).T
        Pw = np.matmul(T, np.matmul(R, Pb))
        Pc = np.matmul(proj, Pw)
        Pc = Pc / Pc[2, :]
        Pcs.append(Pc)
        bounding_boxes.append(min_bounding_box(Pc, crop, image_size, normalize))
        # decrease by 1 so that lables start from 0 
        label = COCO_LABEL[LABELS[bbox[9].astype(np.uint8)] - 1]
        labels.append(label)
    return bounding_boxes, labels, Pcs


def write_label_bbox(train_image_file, label_file):
    """Extract and write label and bbox of training sample to label folder.
    
    Will create directories to label_file if not exists.

    Arg:
        train_image_file: absolute path to training sample
        label_file: absolute path to write yolov3 label txt file
    """

    # Extract label and bbox
    train_bbox_file = train_image_file.replace('image.jpg', 'bbox.bin')
    train_proj_file = train_image_file.replace('image.jpg', 'proj.bin')
    bboxs = np.fromfile(train_bbox_file, dtype=np.float32)
    bboxs = bboxs.reshape([-1, 11])
    proj = np.fromfile(train_proj_file, dtype=np.float32).reshape(3, 4)
    img=mpimg.imread(train_image_file)
    bounding_boxes, labels, _ = get_bounding_boxes(
        bboxs, proj, crop=True, image_size=img.shape[0:2], skip_eval=True, normalize=True)

    # Write to lable folder
    if not os.path.exists(os.path.dirname(label_file)):
        os.makedirs(os.path.dirname(label_file))

    with open(label_file, 'w+') as f:
        for bounding_box, label in zip(bounding_boxes, labels):
            f.writelines('{:d} {:.6f} {:.6f} {:.6f} {:.6f}'.format(
                label,
                bounding_box[0],
                bounding_box[1],
                bounding_box[2],
                bounding_box[3]
            ))

    # Keep track of labeled files
    if len(labels) > 0:
        Labeled_files.append(train_image_file)


if __name__ == '__main__':
    files = glob(os.path.join(TRAINVAL_PATH, '*/*_image.jpg'))
    files.sort()

    print('Extracting and writing bbox and labels to label files.')
    for file in tqdm(files):
        label_file = file.replace('images/', 'labels/').replace('.jpg', '.txt')
        write_label_bbox(file, label_file)

    print('Writing labled files to {}.'.format(Labeled_files_path))
    with open(Labeled_files_path, 'w') as f:
        for file in tqdm(Labeled_files):
            f.writelines('{}\n'.format(file))