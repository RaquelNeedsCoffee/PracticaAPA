"""
This file contains the svm of our credit card data
"""
import pandas as pd
import sklearn.model_selection as ms
import sklearn.neighbors as nb  # knn
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from os import path
from sklearn.svm import SVC
import _pickle as pickle

from Contest.Code.split_data import split


def svm():
    """
    This function reads the data from the split function and calculates the linear svm model
    :return:
    """
    print('We are spliting the data: ')
    (X_train, X_test, X_val, y_train, y_test, y_val) = split(0.999)
    # We scalate the data for using svm
    scaler = MinMaxScaler(feature_range=(-1, 1)).fit(X_train)

    # Apply the normalization trained in training data in both training and test sets
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
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


def main():
    svm()

if __name__ == "__main__":
    main()

