"""
This file is for testing how dummy methods do with the dataset
"""

import sklearn.model_selection as ms
import sklearn.neighbors as nb  # knn
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from os import path
from Contest.Code.split_data import split
import pickle
import gc


def naive_bayes(X_train, X_test, y_train, y_test):
    print('Naive Bayes: ')
    clf = GaussianNB()
    if path.isfile('../Data/Models/knn'):
        clf_trained = pickle.load(open('Data/Models/nb', 'rb'))
    else:
        print('Training the model...')
        clf_trained = clf.fit(X_train, y_train.values.ravel())
        with open('../Data/Models/nb', 'wb') as handle:
            pickle.dump(clf_trained, handle)

    pred = clf_trained.predict(X_test)
    print("Accuracy:", metrics.accuracy_score(y_test, pred))
    print('Full report: \n', metrics.classification_report(y_test, pred))


def knn(X_train, X_test, y_train, y_test):
    print('KNN: ')
    # Create a kNN classifier object
    knc = nb.KNeighborsClassifier()
    if path.isfile('../Data/Models/knn'):
        knc_trained = pickle.load(open('../Data/Models/knn', 'rb'))
    else:
        print('Training the model...')
        # Train the classifier
        knc_trained = knc.fit(X_train, y_train.values.ravel())
        with open('../Data/Models/knn', 'wb') as handle:
            pickle.dump(knc_trained, handle)

    pred = knc_trained.predict(X_test)
    # Obtain accuracy score of learned classifier on test data
    print('Accuracy: ', knc.score(X_test, y_test))
    print('Full report: \n: ', metrics.classification_report(y_test, pred))


def random_forest(X_train, X_test, y_train, y_test):
    print('Random Forest: ')
    rf = RandomForestClassifier(n_estimators=1)
    if path.isfile('../Data/Models/rf'):
        rf_trained = pickle.load(open('Data/Models/rf', 'rb'))
    else:
        print('Training the model...')
        rf_trained = rf.fit(X_train, y_train.values.ravel())
        with open('../Data/Models/rf', 'wb') as handle:
            pickle.dump(rf_trained, handle)

    pred = rf_trained.predict(X_test)
    print("Accuracy:", metrics.accuracy_score(y_test, pred))
    print('Full report: \n', metrics.classification_report(y_test, pred))
    del rf, rf_trained
    gc.collect()


def main():
    (X_train, X_test, X_val, y_train, y_test, y_val) = split(0.95)
    random_forest(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()
