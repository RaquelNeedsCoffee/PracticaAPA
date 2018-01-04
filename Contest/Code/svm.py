"""
This file contains the svm of our credit card data
"""

from sklearn import metrics
from os import path
from sklearn.svm import SVC
import _pickle as pickle
import gc
from Code.trying_mca import data_from_filesMCA

from Code.split_data import split


def linear_svm():
    """
    This function reads the data from the split function and calculates the linear svm model
    :return:
    """
    print('loading data...')
    (X_train, X_test, X_val, y_train, y_test, y_val) = data_from_filesMCA()
    print('data loaded')

    print('linear SVM: ')
    knc = SVC(kernel='linear')
    if path.isfile('../Data/Models/svm.pkl'):
        svm_trained = pickle.load(open('../Data/Models/svm.pkl', 'rb'))
    else:
        print('Training the model...')
        svm_trained = knc.fit(X_train, y_train.values.ravel())
        with open('../Data/Models/svm.pkl', 'wb') as handle:
            pickle.dump(svm_trained, handle)
    pred=svm_trained.predict(X_test)
    print("Confusion matrix on test set:\n", metrics.confusion_matrix(y_test, pred))
    print("\nAccuracy on test set: ", metrics.accuracy_score(y_test, pred))
    print('Full report: \n: ', metrics.classification_report(y_test, pred))
    del svm_trained, knc
    gc.collect()


def quadratic_svm():
    """
    This function reads the data from the split function and calculates the linear svm model
    :return:
    """
    print('We are spliting the data: ')
    (X_train, X_test, X_val, y_train, y_test, y_val) = split(0.999)
    # We scalate the data for using svm

    print('Quadratic SVM: ')
    knc = SVC(kernel='poly', degree=2)
    _path = '../Data/Models/qsvm.pkl'
    if path.isfile(_path):
        svm_trained = pickle.load(open(_path, 'rb'))
    else:
        print('Training the model...')
        svm_trained = knc.fit(X_train, y_train.values.ravel())
        with open(_path, 'wb') as handle:
            pickle.dump(svm_trained, handle)
    pred=svm_trained.predict(X_test)
    print("Confusion matrix on test set:\n", metrics.confusion_matrix(y_test, pred))
    print("\nAccuracy on test set: ", metrics.accuracy_score(y_test, pred))
    print('Full report: \n: ', metrics.classification_report(y_test, pred))
    del svm_trained, knc
    gc.collect()


def rbf_svm(gama):
    """
    This function reads the data from the split function and calculates the linear svm model
    The fit time complexity is more than quadratic with the number of samples which makes
    it hard to scale to dataset with more than a couple of 10000 samples
    :return:
    """
    print('We are spliting the data: ')
    (X_train, X_test, X_val, y_train, y_test, y_val) = split(0.999)

    print('RBF SVM: ')
    knc = SVC(kernel='rbf', gamma=gama)
    _path = '../Data/Models/svm_rbf.pkl'
    if path.isfile(_path):
        svm_trained = pickle.load(open(_path, 'rb'))
    else:
        print('Training the model...')
        svm_trained = knc.fit(X_train, y_train.values.ravel())
        with open(_path, 'wb') as handle:
            pickle.dump(svm_trained, handle)
    pred=svm_trained.predict(X_test)
    print("Confusion matrix on test set:\n", metrics.confusion_matrix(y_test, pred))
    print("\nAccuracy on test set: ", metrics.accuracy_score(y_test, pred))
    print('Full report: \n: ', metrics.classification_report(y_test, pred))
    del svm_trained, knc
    gc.collect()


def main():
    linear_svm()

if __name__ == "__main__":
    main()

