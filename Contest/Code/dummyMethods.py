"""
This file is for testing how dummy methods do with the dataset
"""

import pandas as pd
import sklearn.model_selection as ms
import sklearn.neighbors as nb  # knn
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics


from Code.split_data import split


def main():
    (X_train, X_test, X_val, y_train, y_test, y_val) = split()
    print('KNN: ')
    # Create a kNN classifier object
    knc = nb.KNeighborsClassifier()

    print('Training the model...')
    # Train the classifier
    knc.fit(X_train, y_train.values.ravel())
    pred = knc.fit(X_train, y_train.values.ravel()).predict(X_test)
    # Obtain accuracy score of learned classifier on test data
    # TODO: change accuracy to f mesure
    print('Accuracy: ', knc.score(X_test, y_test))
    print('Full report: \n: ', metrics.classification_report(y_test, pred))

    print('Naive Bayes: ')
    clf = GaussianNB()
    pred = clf.fit(X_train, y_train.values.ravel()).predict(X_test)
    print("Accuracy:", metrics.accuracy_score(y_test, pred))
    print('Full report: \n', metrics.classification_report(y_test, pred))


if __name__ == "__main__":
    main()

