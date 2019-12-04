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
    train_data = np.loadtxt('/Users/cindyshao/Dropbox/ME599/project/trainval_labels_part.csv', skiprows=1, dtype=str, delimiter=',')
    np.random.shuffle(train_data)
    Y = train_data[:, 1].astype(np.int8)
    n = 2000
    Y_train = Y[:n]
    Y_test = Y[n+1:]
    X = train_data[:, 2:]
    X = X.astype(np.float32)
    X = np.multiply(100, X)
    X_train = X[:n,:]
    X_test = X[n+1:,:]
    clf = svm.SVC(decision_function_shape='ovr',gamma='auto')
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