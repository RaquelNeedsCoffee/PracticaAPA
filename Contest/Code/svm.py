"""
This file contains the svm of our credit card data
"""
import pandas as pd
import sklearn.model_selection as ms
import sklearn.neighbors as nb  # knn
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler

from sklearn.svm import SVC


from Code.split_data import split


def svm():
    """
    This function reads the data from the split function and calculates the linear svm model
    :return:
    """
    print('We are spliting the data: ')
    (X_train, X_test, X_val, y_train, y_test, y_val) = split()
    # We scalate the data for using svm
    scaler = MinMaxScaler(feature_range=(-1, 1)).fit(X_train)

    # Apply the normalization trained in training data in both training and test sets
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    print('linear SVM: ')
    knc = SVC(kernel='linear')
    print('Training...')
    knc.fit(X_train, y_train.values.ravel())
    pred=knc.predict(X_test)
    print("Confusion matrix on test set:\n", metrics.confusion_matrix(y_test, pred))
    print("\nAccuracy on test set: ", metrics.accuracy_score(y_test, pred))
    print('Full report: \n: ', metrics.classification_report(y_test, pred))


def main():
    svm()

if __name__ == "__main__":
    main()

