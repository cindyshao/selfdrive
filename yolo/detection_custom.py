import os, zipfile
from gluoncv import utils
import numpy as np
from matplotlib import pyplot as plt
import getopt, sys

DATA_DIR = '/home/ubuntu/data-2019/'
LIST_DIR = '/home/ubuntu/selfdrive/data_lists/'
def rot(n):
    n = np.asarray(n).flatten()
    assert(n.size == 3)

    theta = np.linalg.norm(n)
    if theta:
        n /= theta
        K = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])

        return np.identity(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K
    else:
        return np.identity(3)


def get_bbox(p0, p1):
    """
    Input:
    *   p0, p1
        (3)
        Corners of a bounding box represented in the body frame.

    Output:
    *   v
        (3, 8)
        Vertices of the bounding box represented in the body frame.
    *   e
        (2, 14)
        Edges of the bounding box. The first 2 edges indicate the `front` side
        of the box.
    """
    v = np.array([
        [p0[0], p0[0], p0[0], p0[0], p1[0], p1[0], p1[0], p1[0]],
        [p0[1], p0[1], p1[1], p1[1], p0[1], p0[1], p1[1], p1[1]],
        [p0[2], p1[2], p0[2], p1[2], p0[2], p1[2], p0[2], p1[2]]
    ])
    e = np.array([
        [2, 3, 0, 0, 3, 3, 0, 1, 2, 3, 4, 4, 7, 7],
        [7, 6, 1, 2, 1, 2, 4, 5, 6, 7, 5, 6, 5, 6]
    ], dtype=np.uint8)

    return v, e

def write_line(img_path, im_shape, boxes, ids, idx):
    h, w, c = im_shape
    # for header, we use minimal length 2, plus width and height
    # with A: 4, B: 5, C: width, D: height
    A = 4
    B = 5
    C = w
    D = h
    # concat id and bboxes
    labels = np.hstack((ids.reshape(-1, 1), boxes)).astype('float')
    # normalized bboxes (recommanded)
    labels[:, (1, 3)] /= float(w)
    labels[:, (2, 4)] /= float(h)
    # flatten
    labels = labels.flatten().tolist()
    str_idx = [str(idx)]
    str_header = [str(x) for x in [A, B, C, D]]
    str_labels = [str(x) for x in labels]
    str_path = [img_path]
    line = '\t'.join(str_idx + str_header + str_labels + str_path) + '\n'
    return line


def demo(snapshot):
    proj = np.fromfile(snapshot.replace('_image.jpg', '_proj.bin'), dtype=np.float32)
    proj.resize([3, 4])
    try:
        bbox = np.fromfile(snapshot.replace('_image.jpg', '_bbox.bin'), dtype=np.float32)
    except FileNotFoundError:
        print('[*] bbox not found.')
        bbox = np.array([], dtype=np.float32)

    bbox = bbox.reshape([-1, 11])
    all_boxes = []
    all_labels = []

    for k, b in enumerate(bbox):
        R = rot(b[0:3])
        t = b[3:6]

        sz = b[6:9]
        vert_3D, edges = get_bbox(-sz / 2, sz / 2)
        vert_3D = R @ vert_3D + t[:, np.newaxis]

        vert_2D = proj @ np.vstack([vert_3D, np.ones(vert_3D.shape[1])])
        vert_2D = vert_2D / vert_2D[2, :]

        xmin = int(np.round(min(vert_2D[0,:])))
        xmax = int(np.round(max(vert_2D[0,:])))
        ymin = int(np.round(min(vert_2D[1,:])))
        ymax = int(np.round(max(vert_2D[1, :])))
        box = [xmin, ymin, xmax, ymax]
        all_boxes.append(box)
        label = b[9]
        all_labels.append(label)

    all_boxes = np.asarray(all_boxes)
    all_labels = np.asarray(all_labels)
    return all_boxes, all_labels



def read_data(folder):
    list_data = LIST_DIR + 'labels_' +folder + '.csv'
    test_data = np.loadtxt(list_data, skiprows=1, dtype=str, delimiter=',')
    files = test_data[:, 0]
    write_file = folder + '.txt'
    with open(write_file, 'w') as fw:
        for j, file in enumerate(files):
            name = file + '_image.jpg'
            im_fname = DATA_DIR + 'trainval' + '/'+ name
            img = plt.imread(im_fname)
            all_boxes, all_ids = demo(im_fname)
            for i in range(0, len(all_ids)):
                line = write_line(im_fname, img.shape, all_boxes, all_ids, i)
                fw.write(line)


def main():
    type = sys.argv[1]
    read_data(type)


if __name__ == '__main__':
    main()