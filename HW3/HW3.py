import os
import numpy as np
from PIL import Image
from sklearn import svm
from sklearn.metrics import accuracy_score

def read_data():
    train_path = "./picture/train"
    train_data = []
    train_label = []
    for filename in os.listdir(train_path):
        img_path = os.path.join(train_path, filename)
        img = Image.open(img_path)
        train_data.append(list(img.getdata()))
        file_num = int(filename.split('_')[1].split('.')[0]) - 1
        train_label.append(file_num // 1000)

    test_path = "./picture/test"
    test_data = []
    test_label = []
    for filename in os.listdir(test_path):
        img_path = os.path.join(test_path, filename)
        img = Image.open(img_path)
        test_data.append(list(img.getdata()))
        file_num = int(filename.split('_')[1].split('.')[0]) - 1
        test_label.append(file_num // 500)

    return np.array(train_data), np.array(train_label), np.array(test_data), np.array(test_label)

def CSVM(Cvalue, k, train_data, train_label, test_data, test_label):
    clf = svm.SVC(C=Cvalue, kernel=k)
    clf.fit(train_data, train_label)
    test_pred = clf.predict(test_data)
    print('====================')
    print("C = {0}  Kernel type = {1}".format(Cvalue,k))
    print('accuracy:', accuracy_score(test_label, test_pred))

def NUSVM(NUvalue, k, train_data, train_label, test_data, test_label):
    clf = svm.NuSVC(nu=NUvalue, kernel=k)
    clf.fit(train_data, train_label)
    test_pred = clf.predict(test_data)
    print('====================')
    print("ν = {0}  Kernel type = {1}".format(NUvalue,k))
    print('accuracy:', accuracy_score(test_label, test_pred))

def main():
    train_data, train_label, test_data, test_label = read_data()

    # print('C-SVM')
    # CSVM(0.1, 'rbf', train_data, train_label, test_data, test_label)
    # CSVM(10, 'rbf', train_data, train_label, test_data, test_label)
    # CSVM(1, 'rbf', train_data, train_label, test_data, test_label)
    # CSVM(1, 'linear', train_data, train_label, test_data, test_label)
    # CSVM(1, 'poly', train_data, train_label, test_data, test_label)
    # CSVM(1, 'sigmoid', train_data, train_label, test_data, test_label)

    # print('\nν-SVM')
    # NUSVM(0.1, 'rbf', train_data, train_label, test_data, test_label)
    # NUSVM(0.9, 'rbf', train_data, train_label, test_data, test_label)
    # NUSVM(0.5, 'rbf', train_data, train_label, test_data, test_label)
    # NUSVM(0.5, 'linear', train_data, train_label, test_data, test_label)
    # NUSVM(0.5, 'poly', train_data, train_label, test_data, test_label)
    # NUSVM(0.5, 'sigmoid', train_data, train_label, test_data, test_label)

    clf = svm.SVC()
    clf.fit(train_data, train_label)
    np.set_printoptions(threshold=np.inf, linewidth=np.inf)
    print('\nNumber of support vectors for each class: ', clf.n_support_)
    #print(clf.support_)

    clf = svm.NuSVC()
    clf.fit(train_data, train_label)
    print('\nSupport vectors shape for nu=0.5: ', clf.support_vectors_.shape)

    clf = svm.NuSVC(nu=0.1)
    clf.fit(train_data, train_label)
    print('\nSupport vectors shape for nu=0.1: ', clf.support_vectors_.shape)

    clf = svm.NuSVC(nu=0.9)
    clf.fit(train_data, train_label)
    print('\nSupport vectors shape for nu=0.9: ', clf.support_vectors_.shape)

if __name__ == '__main__':
    main()