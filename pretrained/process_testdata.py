#! /usr/bin/python3
import os
import csv
import numpy as np
from glob import glob

TEST_FILE_PATH = '/Users/cindyshao/Dropbox/ME599/project/rob535-fall-2019-task-1-image-classification/data-2019/test'
OUTPUT_PATH = os.path.dirname(os.path.abspath(__file__))


def write_dummy_labels(test_file_path, output_path):
    """Create dummy label for test samples, just for initial submit."""
    files = glob('{}/*/*_cloud.bin'.format(test_file_path))
    files.sort()
    output_file = '{}/test_nolabels.csv'.format(output_path)
    with open(output_file, 'w') as f:
        writer = csv.writer(f, delimiter=',', lineterminator='\n')
        writer.writerow(['guid/image', 'label'])

        for file in files:
            guid = file.split('/')[-2]
            idx = file.split('/')[-1].replace('_cloud.bin', '')
            label = 0

            writer.writerow(['{}/{}'.format(guid, idx), label])

    print('Wrote report file `{}`'.format(output_file))


if __name__ == '__main__':
    for test_file_path in [TEST_FILE_PATH]:
        write_dummy_labels(test_file_path, OUTPUT_PATH)
