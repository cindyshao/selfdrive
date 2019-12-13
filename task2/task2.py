import os
import numpy as np
from glob import glob
import csv

import d2tod3

DATASET_PATH = '/media/ml/Backup/ROB535_Perception/rob535-fall-2019-task-1-image-classification/images/test'
ALL_BBOX_FILE = os.path.join(DATASET_PATH, 'bbox_sdc_finetune_final.csv')


if __name__ == '__main__':
    # read template.csv to get target image file list
    all_file = []
    with open('template.csv') as f:
        reader = csv.reader(f, delimiter=',')
        # skip header
        next(reader)

        row_num = 0
        for row in reader:
            row_num += 1
            if row_num % 2 == 1:
                guid = row[0].split('/')[0]
                idx = row[0].split('/')[1]
                all_file.append(os.path.join(DATASET_PATH, '{}/{}_image.jpg'.format(guid, idx)))

    # # read all boxes
    all_box = {}
    with open(ALL_BBOX_FILE) as f:
        reader = csv.reader(f, delimiter=',')
        # skip header
        next(reader)

        for row in reader:
            key = os.path.join(DATASET_PATH, '{}_image.jpg'.format(row[0]))
            bbox = [int(i) for i in row[1:5]]
            all_box.update({key : bbox})

    print(len(all_file))
    print(len(all_box))

    with open('task2.csv', 'w') as f:
        writer = csv.writer(f, delimiter=',', lineterminator='\n')
        writer.writerow(['guid/image/axis', 'value'])

        for file in all_file:
            box = all_box[file]
            r, theta = d2tod3.getcenter(file, box)
            theta = theta*180/np.pi
            print(r, theta)
            guid = file.split('/')[-2]
            idx = file.split('/')[-1].replace('_image.jpg', '')
            writer.writerow(['{}/{}/r'.format(guid, idx), r if not np.isnan(r) else 50])
            writer.writerow(['{}/{}/theta'.format(guid, idx), theta if not np.isnan(theta) else 0])
