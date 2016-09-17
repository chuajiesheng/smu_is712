import numpy as np
from sklearn import svm
from sklearn import linear_model
from sklearn import naive_bayes
from sklearn import ensemble
import random
import sys
from scipy import sparse
import time
from sklearn.datasets import load_svmlight_file


def feature_norm(X):
    mu = np.mean(X, 0)
    sigma = np.std(X, 0)
    X_norm = (X - mu) / sigma
    return X_norm


def importdata(path):
    fpr = open(path, 'r')
    X = []
    for line in fpr:
        vals = line.strip().split(' ')
        fea = [float(val) for val in vals]
        X.append(fea)
    fpr.close()
    return X


def data_split(data_name, sparse_bool):
    if data_name == 'a7a':
        X_train, Y_train = load_svmlight_file("../data/a7a.train")
        X_train = X_train.toarray()
        fea_num = X_train.shape[1]

        X_test, Y_test = load_svmlight_file("../data/a7a.test")
        X_test = X_test.toarray()

        if X_test.shape[1] > fea_num:
            X_test = X_test[:, :fea_num]

    elif data_name == 'gisette':
        X_train = importdata("../data/gisette_train.data")
        X_train = np.array(X_train) / 1000
        Y_train = importdata("../data/gisette_train.labels")
        Y_train = (np.array(Y_train)[:, 0] + 1) / 2

        X_test = importdata("../data/gisette_test.data")
        X_test = np.array(X_test) / 1000;
        Y_test = importdata("../data/gisette_test.labels")
        Y_test = (np.array(Y_test)[:, 0] + 1) / 2

    else:
        print ('no existing data!!!')

    if sparse_bool == 'sparse':
        X_train = sparse.csr_matrix(X_train)
        X_test = sparse.csr_matrix(X_test)

    return X_train, Y_train, X_test, Y_test


def model_test(X_test, Y_test, svr):
    predictY = svr.predict(X_test)
    result = sum(predictY == Y_test)

    return float(result) / len(Y_test)


# You may add more models here
def model_train(X_train, Y_train, model_name, para_c=1, para_g=0):
    if model_name == 'svm_kernel':
        svr = svm.SVC(C=para_c, gamma=para_g, kernel='rbf')
    elif model_name == 'svm_linear':
        svr = svm.LinearSVC(C=para_c, penalty='l2')
    elif model_name == 'logisticR':
        svr = linear_model.LogisticRegression(C=para_c, penalty='l2')
    elif model_name == 'gaussian':
        svr = naive_bayes.GaussianNB()
    elif model_name == 'ada':
        svr = ensemble.AdaBoostClassifier()
    else:
        print ('no existing model!!')

    svr.fit(X_train, Y_train)

    return svr


def model_training(X_train, Y_train, model_name):
    '''
    Your code here for model selection / hyperparameter selection
    Return the model with the best hyper-parameter (e.g., k-fold Cross validation)
    '''
    # svr = model_train(X_train, Y_train, model_name)

    return svr


def main():
    # command: python main.py heart svm_linear sparce
    data_name = sys.argv[1]  # heart or gisette
    model_name = sys.argv[2]  # svm_linear or logisticR or svm_kernel
    sparse_bool = sys.argv[3]  # sparse or not

    X_train, Y_train, X_test, Y_test = data_split(data_name, sparse_bool)

    start_time = time.time()
    svr = model_training(X_train, Y_train, model_name)
    print("Tuning time: %s seconds" % (time.time() - start_time))

    accuracy_train = model_test(X_train, Y_train, svr)
    accuracy_test = model_test(X_test, Y_test, svr)
    print ("training data set accuracy:" + str(accuracy_train))
    print ("test data set accuracy:" + str(accuracy_test))


if __name__ == '__main__':
    main()
