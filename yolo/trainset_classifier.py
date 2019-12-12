from sklearn import svm
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
# from gluoncv import model_zoo, data, utils
from matplotlib import pyplot as plt
import numpy as np
from glob import glob
import csv
import os
import pickle

def main():
    train_data = np.loadtxt('/Users/cindyshao/Dropbox/ME599/train_labels.csv', skiprows=1, dtype=str, delimiter=',')
    test_data = np.loadtxt('/Users/cindyshao/Dropbox/ME599/val_labels.csv', skiprows=1, dtype=str, delimiter=',')
    np.random.shuffle(train_data)
    np.random.shuffle(test_data)
    Y_train = train_data[:, 1].astype(np.int8)
    Y_test = test_data[:, 1].astype(np.int8)
    X_train = train_data[:, 2:].astype(np.float32)
    X_train = np.multiply(1000, X_train)
    X_test = test_data[:, 2:].astype(np.float32)
    X_test = np.multiply(1000, X_test)


    clf = svm.SVC(decision_function_shape='ovo',gamma='auto')
    # clf = tree.DecisionTreeClassifier()
    # clf = AdaBoostClassifier(n_estimators=10)
    # clf = RandomForestClassifier(n_estimators=10)
    # clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=1000)
    clf.fit(X_train, Y_train)

    # accuracy1=clf.score(X_train, Y_train)
    # print(accuracy1)
    Y_pred = clf.predict(X_test)
    diff = np.subtract(Y_test, Y_pred)
    wrong = sum(diff != 0)
    accuracy = 1 - wrong/len(Y_test)
    print(accuracy)
    filename = 'finalized_model.sav'
    pickle.dump(clf, open(filename, 'wb'))

if __name__ == "__main__":
    main()