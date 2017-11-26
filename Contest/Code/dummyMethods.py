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
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier

import pickle
import gc


def lda(X_train, X_test, y_train, y_test):
    """
    LDA
    Suposiciones:
    - Gausianidad de los datos
    - La misma varianza
    Hiperparametros: No
    :param X_train:
    :param X_test:
    :param y_train:
    :param y_test:
    :return:
    """
    print("LDA:")
    # Importante estandarizar datos
    lda = LinearDiscriminantAnalysis()
    _path = '../Data/Models/lda.plk'
    if path.isfile(_path):
        lda_trained = pickle.load(open(_path, 'rb'))
    else:
        print('Training the model...')
        lda_trained = lda.fit(X_train, y_train.values.ravel())
        with open(_path, 'wb') as handle:
            pickle.dump(lda_trained, handle)
    pred = lda_trained.predict(X_test)
    print("Accuracy:", metrics.accuracy_score(y_test, pred))
    print('Full report: \n', metrics.classification_report(y_test, pred))
    del lda, lda_trained
    gc.collect()


def qda(X_train, X_test, y_train, y_test):
    """
    QDA
    Suposiciones:
    - Gausianidad de los datos
    - Diferentes varianzas
    Hiperparametros:
    - No
    :param X_train:
    :param X_test:
    :param y_train:
    :param y_test:
    :return:
    """
    print("QDA:")
    # Importante estandarizar datos
    qda = QuadraticDiscriminantAnalysis()
    _path = '../Data/Models/qda.plk'
    if path.isfile(_path):
        qda_trained = pickle.load(open(_path, 'rb'))
    else:
        print('Training the model...')
        qda_trained = qda.fit(X_train, y_train.values.ravel())
        with open(_path, 'wb') as handle:
            pickle.dump(qda_trained, handle)
    pred = qda_trained.predict(X_test)
    print("Accuracy:", metrics.accuracy_score(y_test, pred))
    print('Full report: \n', metrics.classification_report(y_test, pred))
    del qda, qda_trained
    gc.collect()


def rda(X_train, X_test, y_train, y_test, reg):
    """
    RDA
    Suposiciones:
    - Gausianidad de los datos
    - Diferentes varianzas
    Hiperparametros:
    - regularizrion parameter
    :param X_train:
    :param X_test:
    :param y_train:
    :param y_test:
    :return:
    """
    print("RDA:")
    # Importante estandarizar datos
    rda = QuadraticDiscriminantAnalysis(reg_param=reg)
    _path = '../Data/Models/rda.plk'
    if path.isfile(_path):
        rda_trained = pickle.load(open(_path, 'rb'))
    else:
        print('Training the model...')
        rda_trained = rda.fit(X_train, y_train.values.ravel())
        with open(_path, 'wb') as handle:
            pickle.dump(rda_trained, handle)
    pred = rda_trained.predict(X_test)
    print("Accuracy:", metrics.accuracy_score(y_test, pred))
    print('Full report: \n', metrics.classification_report(y_test, pred))
    del rda, rda_trained
    gc.collect()


def logistic_regresion(X_train, X_test, y_train, y_test):
    """
    La regresión logística de toda la vida.
    La idea es hacer un clasificador lineal y pasarlo por la función logística.
    P(C|x) = g(wTx+w0)
    Suposiciones:
    -
    Hiperparametros:
    - No
    :param X_train:
    :param X_test:
    :param y_train:
    :param y_test:
    :return:
    """
    print("Regresion Lineal:")
    # Importante estandarizar datos
    lr = LogisticRegression(solver='saga', n_jobs=-1)
    _path = '../Data/Models/lr.plk'
    if path.isfile(_path):
        lr_trained = pickle.load(open(_path, 'rb'))
    else:
        print('Training the model...')
        lr_trained = lr.fit(X_train, y_train.values.ravel())
        with open(_path, 'wb') as handle:
            pickle.dump(lr_trained, handle)
    pred = lr_trained.predict(X_test)
    print("Accuracy:", metrics.accuracy_score(y_test, pred))
    print('Full report: \n', metrics.classification_report(y_test, pred))
    del lr, lr_trained
    gc.collect()


def naive_bayes(X_train, X_test, y_train, y_test):
    """
    Hiperparametros:
    - No
    :param X_train:
    :param X_test:
    :param y_train:
    :param y_test:
    :return:
    """
    print('Naive Bayes: ')
    clf = GaussianNB()
    if path.isfile('../Data/Models/nb'):
        clf_trained = pickle.load(open('../Data/Models/nb', 'rb'))
    else:
        print('Training the model...')
        clf_trained = clf.fit(X_train, y_train.values.ravel())
        with open('../Data/Models/nb', 'wb') as handle:
            pickle.dump(clf_trained, handle)

    pred = clf_trained.predict(X_test)
    print("Accuracy:", metrics.accuracy_score(y_test, pred))
    print('Full report: \n', metrics.classification_report(y_test, pred))
    del clf, clf_trained
    gc.collect()


def knn(X_train, X_test, y_train, y_test, neighbors):
    """
    Hiperparametros:
    - n_neighbours
    :param X_train:
    :param X_test:
    :param y_train:
    :param y_test:
    :param n_neighbors:
    :return:
    """
    print('KNN ', neighbors, ' :')
    # Create a kNN classifier object
    knc = nb.KNeighborsClassifier(n_neighbors=neighbors, n_jobs=-1)
    _path = '../Data/Models/knn' + str(neighbors) + 'pkl'
    if path.isfile(_path):
        knc_trained = pickle.load(open(_path, 'rb'))
    else:
        print('Training the model...')
        # Train the classifier
        knc_trained = knc.fit(X_train, y_train.values.ravel())
        with open(_path, 'wb') as handle:
            pickle.dump(knc_trained, handle)
        print('trained')

    pred = knc_trained.predict(X_test)
    # Obtain accuracy score of learned classifier on test data
    print('Accuracy: ', knc.score(X_test, y_test))
    print('Full report: \n: ', metrics.classification_report(y_test, pred))
    del knc_trained, knc
    gc.collect()


def random_forest(X_train, X_test, y_train, y_test, n_estimators):
    print('Random Forest ', n_estimators, ' :')
    rf_path = '../Data/Models/rf' + str(n_estimators) + '.pkl'
    rf = RandomForestClassifier(n_estimators=n_estimators)
    if path.isfile(rf_path):
        rf_trained = pickle.load(open(rf_path, 'rb'))
    else:
        print('Training the model...')
        rf_trained = rf.fit(X_train, y_train.values.ravel())
        with open(rf_path, 'wb') as handle:
            pickle.dump(rf_trained, handle)

    pred = rf_trained.predict(X_test)
    print("Accuracy:", metrics.accuracy_score(y_test, pred))
    print('Full report: \n', metrics.classification_report(y_test, pred))
    del rf, rf_trained
    gc.collect()


def perceptron(X_train, X_test, X_val, y_train, y_test, y_val, reg, pen):
    """
    Hiperparametros:
    - pen (l1 o l2 o elasticnet) norma de regularizacion
    - reg (alpha en (0,1))constante de regularizacion
    :param X_train:
    :param X_test:
    :param X_val:
    :param y_train:
    :param y_test:
    :param y_val:
    :param reg:
    :return:
    """
    print('Perceptron : ')
    _path = '../Data/Models/perceptron.pkl'
    per = Perceptron(penalty=pen, alpha=reg, n_jobs=-1)
    if path.isfile(_path):
        per_trained = pickle.load(open(_path, 'rb'))
    else:
        print('Training the model...')
        per_trained = per.fit(X_train, y_train.values.ravel())
        with open(_path, 'wb') as handle:
            pickle.dump(per_trained, handle)

    pred = per_trained.predict(X_test)
    print("Accuracy:", metrics.accuracy_score(y_test, pred))
    print('Full report: \n', metrics.classification_report(y_test, pred))
    del per, per_trained
    gc.collect()


def mlp(X_train, X_test, X_val, y_train, y_test, y_val, reg, size=(1, 20), act='logistic'):
    """
    Hiperparametros:
    - reg el parametro de recularizacion
    - size el tamaño de los hiden layers
    - act {‘identity’, ‘logistic’, ‘tanh’, ‘relu’} la funcion de activación
    :param X_train:
    :param X_test:
    :param X_val:
    :param y_train:
    :param y_test:
    :param y_val:
    :param reg:
    :param size:
    :param act:
    :return:
    """
    print('Multi Layer Perceptron : ')
    _path = '../Data/Models/mlp.pkl'
    mlp = MLPClassifier(hidden_layer_sizes=size, activation=act, alpha=reg, verbose=True)
    if path.isfile(_path):
        mlp_trained = pickle.load(open(_path, 'rb'))
    else:
        print('Training the model...')
        mlp_trained = mlp.fit(X_train, y_train.values.ravel())
        print('Trained')
        with open(_path, 'wb') as handle:
            pickle.dump(mlp_trained, handle)
    pred = mlp_trained.predict(X_test)
    print("Accuracy:", metrics.accuracy_score(y_test, pred))
    print('Full report: \n', metrics.classification_report(y_test, pred))
    del mlp, mlp_trained, pred
    gc.collect()


def run_rf_knn(X_train, X_test, X_val, y_train, y_test, y_val):
    for i in [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]:
        # random_forest(X_train, X_test, y_train, y_test, i)
        knn(X_train, X_test, y_train, y_test, i)


def main():
    split_range = 0.2
    print('El split range es: ', split_range)
    (X_train, X_test, X_val, y_train, y_test, y_val) = split(split_range)
    # naive_bayes(X_train, X_test, y_train, y_test)
    # lda(X_train, X_test, y_train, y_test)
    # qda(X_train, X_test, y_train, y_test)
    # logistic_regresion(X_train, X_test, y_train, y_test)
    run_rf_knn(X_train, X_test, X_val, y_train, y_test, y_val)


if __name__ == "__main__":
    main()
