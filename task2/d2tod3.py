import numpy as np
from matplotlib import pyplot as plt
import getopt, sys


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

def getbox(snapshot):
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

        xmin = int(np.round(min(vert_2D[0, :])))
        xmax = int(np.round(max(vert_2D[0, :])))
        ymin = int(np.round(min(vert_2D[1, :])))
        ymax = int(np.round(max(vert_2D[1, :])))
        box = [xmin, ymin, xmax, ymax]
        all_boxes.append(box)
        label = b[9]
        if label == 22:
            label = 0
        elif label == 21:
            label = 18
        elif label == 20:
            label = 19
        all_labels.append(label)

    all_boxes = np.asarray(all_boxes)
    all_labels = np.asarray(all_labels)
    return all_boxes, all_labels

def getcenter(snapshot, box):
    xyz = np.fromfile(snapshot.replace('_image.jpg', '_cloud.bin'), dtype=np.float32)
    xyz = xyz.reshape([3, -1])
    proj = np.fromfile(snapshot.replace('_image.jpg', '_proj.bin'), dtype=np.float32)
    proj.resize([3, 4])
    uv = proj @ np.vstack([xyz, np.ones_like(xyz[0, :])])
    uv = uv / uv[2, :]
    xmin, ymin, xmax, ymax = box
    a1 = uv[0, :] > xmin
    a2 = uv[0, :] < xmax
    a3 = uv[1, :] > ymin
    a4 = uv[1, :] < ymax
    mask1 = np.logical_and(a1,a2)
    mask2 =  np.logical_and(a3,a4)
    mask = np.logical_and(mask1,mask2)
    part = xyz[:,mask]
    vec = np.median(part, 1)
    # print(vec) # [x,y,z]
    r = np.linalg.norm(vec)
    theta = np.arctan(vec[0]/vec[2])
    return r, theta


def main():
    file = '/media/ml/Backup/ROB535_Perception/rob535-fall-2019-task-1-image-classification/images/trainval/0a4dd0b3-e034-48c1-aab5-023e22d6a73c/0002_image.jpg'
    all_boxes, all_labels = getbox(file)
    box = all_boxes[0]
    r, theta = getcenter(file, box)
    print(r)
    print(theta)


if __name__ == '__main__':
    main()