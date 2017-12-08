from sklearn.neighbors import KNeighborsClassifier  # knn
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier

from os import path
from os import mkdir

from Code.split_data import split
from Code.models import lda, qda, val_rda, logistic_regresion, naive_bayes, random_forest, val_knn
import pickle
import gc
split_range = 0.5
global_path = '../Data/Models/' + str(split_range) + '/'

def main():

	if not path.isdir(global_path):
		mkdir(global_path)
	print('El split range es: ', split_range)

	(X_train, X_test, X_val, y_train, y_test, y_val) = split(split_range)
	lda(X_train, X_test, y_train, y_test)
	qda(X_train, X_test, y_train, y_test)
	val_rda(X_train, X_test, X_val, y_train, y_test, y_val)
	naive_bayes(X_train, X_test, y_train, y_test)
	logistic_regresion(X_train, X_test, y_train, y_test)
	val_knn(X_train, X_test, X_val, y_train, y_test, y_val)

if __name__ == "__main__":
	main()
