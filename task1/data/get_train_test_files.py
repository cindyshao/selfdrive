"""Script to generate file specifying paths to train set."""
import os
import csv
import numpy as np
from glob import glob
import random

DATASET_ROOTDIR = '/media/ml/Backup/ROB535_Perception/rob535-fall-2019-task-1-image-classification/'
TRAINED_FILE_PATH = os.path.join(DATASET_ROOTDIR, 'images/trainval/')
LABELED_TRAINVAL_FILES = os.path.join(DATASET_ROOTDIR, 'labeled_trainval_files.txt')

OUTPUT_PATH = DATASET_ROOTDIR
Train_file_name = 'train_location.txt'
Val_file_name = 'val_location.txt'


def write_photo_location(
    trainval_files, writen_file_path, train_output_path, val_output_path, 
    split=0.1, rand_seed=0):
    """Write and split training photo paths to their output_path."""

    if trainval_files is None:
        trainval_files = glob('{}/*/*_image.jpg'.format(writen_file_path))
        trainval_files.sort()

    # trainval_files = trainval_files[:200]

    np.random.seed(rand_seed)
    np.random.shuffle(trainval_files)

    threshold_idx = int(split * len(trainval_files))
    with open(train_output_path, 'w+') as f:
        for file in trainval_files[0:-threshold_idx]:
            f.writelines('{}\n'.format(file))
    print('Wrote file `{}`'.format(train_output_path))

    with open(val_output_path, 'w+') as f:
        # for file in trainval_files[-threshold_idx:]:
        for file in trainval_files[:threshold_idx]:
            f.writelines('{}\n'.format(file))
    print('Wrote file `{}`'.format(val_output_path))


if __name__ == '__main__':
    labeled_trainval_files = [
        line.rstrip('\n') for line in open(LABELED_TRAINVAL_FILES, 'r')]

    write_photo_location(
        writen_file_path=TRAINED_FILE_PATH,
        trainval_files=None,
        train_output_path=os.path.join(OUTPUT_PATH, Train_file_name),
        val_output_path=os.path.join(OUTPUT_PATH, Val_file_name),
        split=0.1
    )


