from sklearn import svm
# from gluoncv import model_zoo, data, utils
from matplotlib import pyplot as plt
import numpy as np
from glob import glob
import csv
import os
import pickle


OUTPUT_PATH = os.path.dirname(os.path.abspath(__file__))


def write_test_labels(output_path):
    filename = 'finalized_model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    test_data = np.loadtxt('/Users/cindyshao/Dropbox/ME599/project/test_labels.csv', skiprows=1, dtype=str,
                            delimiter=',')
    files = test_data[:, 0]
    X = test_data[:, 2:]
    X = X.astype(np.float32)
    X = np.multiply(100, X)
    Y_pred = loaded_model.predict(X)
    output_file = '{}/test_submit_labels.csv'.format(output_path)
    with open(output_file, 'w') as f:
        writer = csv.writer(f, delimiter=',', lineterminator='\n')
        writer.writerow(['guid/image', 'label'])
        for i, file in enumerate(files):
            label = Y_pred[i]
            writer.writerow([file, label])

    print('Wrote report file `{}`'.format(output_file))


if __name__ == '__main__':
    write_test_labels(OUTPUT_PATH)
