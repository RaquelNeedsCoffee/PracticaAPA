from os import path
from os import mkdir
from time import time
from Code.split_data import split
from Code.trying_mca import data_from_files
from Code.models import lda, qda, val_rda, logistic_regresion, naive_bayes, val_rf, val_knn, val_perceptron
import pickle
import gc

split_range = 0.5
global_path = '../Data/Models/' + str(split_range) + '/'


def main():
	init = time()
	if not path.isdir(global_path):
		mkdir(global_path)
	print('El split range es: ', split_range)
	print('loading data...')
	(X_train, X_test, X_val, y_train, y_test, y_val) = data_from_files()
	print('data loaded')
	# lda(X_train, X_test, y_train, y_test)
	# print('lda time', time() -init)
	# init = time()
	# qda(X_train, X_test, y_train, y_test)
	# print('qda time', time() - init)
	# init = time()
	# val_rda(X_train, X_test, X_val, y_train, y_test, y_val)
	# print('rda time', time() - init)
	# init = time()
	# naive_bayes(X_train, X_test, y_train, y_test)
	# print('naive time', time() - init)
	# init = time()
	# logistic_regresion(X_train, X_test, y_train, y_test)
	# print('log reg time', time() - init)
	# init = time()
	# val_knn(X_train, X_test, X_val, y_train, y_test, y_val)
	# print('knn time', time() - init)
	# init = time()
	# val_rf(X_train, X_test, X_val, y_train, y_test, y_val)
	# print('rf time', time() - init)
	init = time()
	val_perceptron(X_train, X_test, X_val, y_train, y_test, y_val)
	print('perceptron time', time() - init)


if __name__ == "__main__":
	main()
