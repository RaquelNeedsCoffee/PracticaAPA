"""
This file is for testing how dummy methods do with the dataset
"""

import sklearn.model_selection as ms
from sklearn.neighbors import KNeighborsClassifier  # knn
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint
from scipy.stats import uniform

import numpy as np
from os import path
import pickle
import gc

from Code.split_data import split

split_range = 0.5
global_path = '../Data/Models/' + str(split_range) + '/'


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
	_report_path = global_path + 'lda_report' + '.txt'
	repport = open(_report_path, 'w')
	print("LDA:")
	# Importante estandarizar datos
	lda = LinearDiscriminantAnalysis()
	_path = global_path + 'lda.plk'
	if path.isfile(_path):
		lda_trained = pickle.load(open(_path, 'rb'))
	else:
		print('Training the model...')
		lda_trained = lda.fit(X_train, y_train.values.ravel())
		with open(_path, 'wb') as handle:
			pickle.dump(lda_trained, handle)
	pred = lda_trained.predict(X_test)
	acc = metrics.accuracy_score(y_test, pred)
	print("Accuracy:", acc)
	print('Full report: \n', metrics.classification_report(y_test, pred))
	repport.write('LDA with test' + ' :' + '\n')
	repport.write("Accuracy:" + str(acc) + '\n')
	repport.write('Full report: \n' + str(metrics.classification_report(y_test, pred)) + '\n')
	del lda, lda_trained
	repport.close()
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
	_report_path = global_path + 'qda_report' + '.txt'
	repport = open(_report_path, 'w')
	print("QDA:")
	# Importante estandarizar datos
	qda = QuadraticDiscriminantAnalysis()
	_path = global_path + 'qda.plk'
	if path.isfile(_path):
		qda_trained = pickle.load(open(_path, 'rb'))
	else:
		print('Training the model...')
		qda_trained = qda.fit(X_train, y_train.values.ravel())
		with open(_path, 'wb') as handle:
			pickle.dump(qda_trained, handle)
	pred = qda_trained.predict(X_test)
	acc = metrics.accuracy_score(y_test, pred)
	print("Accuracy:", acc)
	print('Full report: \n', metrics.classification_report(y_test, pred))
	repport.write('QDA with test' + ' :' + '\n')
	repport.write("Accuracy:" + str(acc) + '\n')
	repport.write('Full report: \n' + str(metrics.classification_report(y_test, pred)) + '\n')
	del qda, qda_trained
	repport.close()
	gc.collect()


def val_rda(X_train, X_test, X_val, y_train, y_test, y_val):
	reg = np.arange(0.01, 2.0, 0.1)
	_report_path = global_path + 'rda_report' + '.txt'
	repport = open(_report_path, 'w')
	all_accuracies = []
	best_index = 0
	i = 0
	for r in reg:
		rda = QuadraticDiscriminantAnalysis(reg_param=r)
		_path = global_path + 'rda_' + str(r) + '.plk'
		if path.isfile(_path):
			rda_trained = pickle.load(open(_path, 'rb'))
		else:
			print('Training the model...')
			rda_trained = rda.fit(X_train, y_train.values.ravel())
			with open(_path, 'wb') as handle:
				pickle.dump(rda_trained, handle)
		pred = rda_trained.predict(X_val)
		acc = metrics.accuracy_score(y_val, pred)
		all_accuracies.append(acc)
		if acc > all_accuracies[best_index]:
			best_index = i
		repport.write('RDA ' + str(r) + ' :' + '\n')
		repport.write("Accuracy:" + str(acc) + '\n')
		repport.write('Full report: \n' + str(metrics.classification_report(y_val, pred)) + '\n')
		del rda, rda_trained
		i += 1
	repport.write('In the test set with the best param we have: \n')
	_path = global_path + 'rda_' + str(reg[best_index]) + '.plk'
	rda_trained = pickle.load(open(_path, 'rb'))
	pred = rda_trained.predict(X_test)
	acc = metrics.accuracy_score(y_test, pred)
	repport.write("Accuracy:" + str(acc) + '\n')
	repport.write('Full report: \n' + str(metrics.classification_report(y_test, pred)) + '\n')
	repport.close()


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
	_path = global_path + 'rda' + reg + '.plk'
	if path.isfile(_path):
		rda_trained = pickle.load(open(_path, 'rb'))
	else:
		print('Training the model...')
		rda_trained = rda.fit(X_train, y_train.values.ravel())
		with open(_path, 'wb') as handle:
			pickle.dump(rda_trained, handle)
	pred = rda_trained.predict(X_test)
	acc = metrics.accuracy_score(y_test, pred)
	print("Accuracy:", acc)
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
	_report_path = global_path + 'logreg_report' + '.txt'
	repport = open(_report_path, 'w')
	print("Regresion Lineal:")
	# Importante estandarizar datos
	lr = LogisticRegression(solver='saga', n_jobs=-1)
	_path = global_path + 'lr.plk'
	if path.isfile(_path):
		lr_trained = pickle.load(open(_path, 'rb'))
	else:
		print('Training the model...')
		lr_trained = lr.fit(X_train, y_train.values.ravel())
		with open(_path, 'wb') as handle:
			pickle.dump(lr_trained, handle)
	pred = lr_trained.predict(X_test)
	acc = metrics.accuracy_score(y_test, pred)
	print("Accuracy:", acc)
	print('Full report: \n', metrics.classification_report(y_test, pred))
	repport.write('Logistic regression with test ' + ' :' + '\n')
	repport.write("Accuracy:" + str(acc) + '\n')
	repport.write('Full report: \n' + str(metrics.classification_report(y_test, pred)) + '\n')

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
	_report_path = global_path + 'naive_report' + '.txt'
	repport = open(_report_path, 'w')
	print('Naive Bayes: ')
	clf = GaussianNB()
	if path.isfile(global_path + 'nb'):
		clf_trained = pickle.load(open(global_path + 'nb', 'rb'))
	else:
		print('Training the model...')
		clf_trained = clf.fit(X_train, y_train.values.ravel())
		with open(global_path + 'nb', 'wb') as handle:
			pickle.dump(clf_trained, handle)

	pred = clf_trained.predict(X_test)
	acc = metrics.accuracy_score(y_test, pred)
	met = metrics.classification_report(y_test, pred)
	print("Accuracy:", acc)
	print('Full report: \n', met)
	repport.write('Logistic regression with test ' + ' :' + '\n')
	repport.write("Accuracy:" + str(acc) + '\n')
	repport.write('Full report: \n' + str(met) + '\n')
	del clf, clf_trained
	gc.collect()


def val_knn(X_train, X_test, X_val, y_train, y_test, y_val):
	kvalues = np.arange(1, 15, 2)
	_report_path = global_path + 'knn_report' + '.txt'
	repport = open(_report_path, 'w')
	all_accuracies = []
	best_index = 0
	i = 0
	for k in kvalues:
		knc = KNeighborsClassifier(algorithm='ball_tree', n_neighbors=k, n_jobs=-1)
		_path = global_path + 'knn_' + str(k) + '.plk'
		if path.isfile(_path):
			knc_trained = pickle.load(open(_path, 'rb'))
		else:
			print('Training the model...')
			knc_trained = knc.fit(X_train, y_train.values.ravel())
			with open(_path, 'wb') as handle:
				pickle.dump(knc_trained, handle)
		pred = knc_trained.predict(X_val)
		acc = metrics.accuracy_score(y_val, pred)
		met = metrics.classification_report(y_val, pred)
		all_accuracies.append(acc)
		if acc > all_accuracies[best_index]:
			best_index = i
		print('KNN with val and k = ' + str(k) + ' :' + '\n')
		print("Accuracy:" + str(acc) + '\n')
		print('Full report: \n' + str(met) + '\n')
		repport.write('KNN with val and k = ' + str(k) + ' :' + '\n')
		repport.write("Accuracy:" + str(acc) + '\n')
		repport.write('Full report: \n' + str(met) + '\n')
		del knc_trained, knc
		i += 1
	repport.write('In the test set with the best param we have: \n')
	_path = global_path + 'knn_' + str(kvalues[best_index]) + '.plk'
	knc_trained = pickle.load(open(_path, 'rb'))
	pred = knc_trained.predict(X_test)
	acc = metrics.accuracy_score(y_test, pred)
	met = metrics.classification_report(y_test, pred)
	repport.write("Accuracy:" + str(acc) + '\n')
	repport.write('Full report: \n' + str(met) + '\n')
	repport.close()


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
	knc = KNeighborsClassifier(algorithm='ball_tree', n_neighbors=neighbors, n_jobs=-1)
	_path = global_path + 'knn' + str(neighbors) + 'ball_tree' + 'pkl'
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


def report(results, n_top=3):
	for i in range(1, n_top + 1):
		candidates = np.flatnonzero(results['rank_test_score'] == i)
		for candidate in candidates:
			print("Model with rank: {0}".format(i))
			print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
				results['mean_test_score'][candidate],
				results['std_test_score'][candidate]))
			print("Parameters: {0}".format(results['params'][candidate]))
			print("")


def val_model(model = RandomForestClassifier()):
	param_dist = {"max_depth": [3, None],
	              "max_features": sp_randint(1, 11),
	              "min_samples_split": sp_randint(2, 11),
	              "min_samples_leaf": sp_randint(1, 11),
	              "bootstrap": [True, False],
	              "criterion": ["gini", "entropy"],
	              "n_estimators": sp_randint(10, 20)}
	# run randomized search
	n_iter_search = 20
	random_search = RandomizedSearchCV(model, param_distributions=param_dist,
	                                   n_iter=n_iter_search)


def val_rf(X_train, X_test, X_val, y_train, y_test, y_val):
	n_estimators = np.arange(10, 20, 2)
	criterion = ["gini", "entropy"]
	_report_path = global_path + 'rf_reportdeph' + '.txt'
	repport = open(_report_path, 'w')
	all_accuracies = []
	best_index = 0
	best_criterion = criterion[0]
	i = 0
	for k in n_estimators:
		for c in criterion:
			knc = RandomForestClassifier(n_estimators = k,criterion=c, n_jobs=-1)
			_path = global_path + 'rfd_' + str(k) + c +'.plk'
			if path.isfile(_path):
				knc_trained = pickle.load(open(_path, 'rb'))
			else:
				print('Training the model...')
				knc_trained = knc.fit(X_train, y_train.values.ravel())
				with open(_path, 'wb') as handle:
					pickle.dump(knc_trained, handle)
			pred = knc_trained.predict(X_val)
			acc = metrics.accuracy_score(y_val, pred)
			met = metrics.classification_report(y_val, pred)
			all_accuracies.append(acc)
			if acc > all_accuracies[best_index]:
				best_index = i
				best_criterion = c
			repport.write('Random Forest with val and k = ' + str(k) +'Criterion '+str(c) + ' :' + '\n')
			repport.write("Accuracy:" + str(acc) + '\n')
			repport.write('Full report: \n' + str(met) + '\n')
			del knc_trained, knc
		i += 1
	repport.write('In the test set with the best param we have: \n')
	_path = global_path + 'rfd_' + str(n_estimators[best_index]) + best_criterion + '.plk'
	rf_trained = pickle.load(open(_path, 'rb'))
	pred = rf_trained.predict(X_test)
	acc = metrics.accuracy_score(y_test, pred)
	met = metrics.classification_report(y_test, pred)
	repport.write("Accuracy:" + str(acc) + '\n')
	repport.write('Full report: \n' + str(met) + '\n')
	repport.close()


def random_forest(X_train, X_test, y_train, y_test, n_estimators):
	print('Random Forest ', n_estimators, ' :')
	rf_path = global_path + 'rf' + str(n_estimators) + '.pkl'
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


def val_perceptron(X_train, X_test, X_val, y_train, y_test, y_val):
	alphas = np.arange(0.01, 2.0, 0.1)
	penalty = ["l1", "l2",'elasticnet']
	_report_path = global_path + 'perceptron_report' + '.txt'
	repport = open(_report_path, 'w')
	all_accuracies = []
	best_index = 0
	best_penalty = penalty[0]
	i = 0
	for k in alphas:
		for c in penalty:
			per = Perceptron(penalty=c, alpha=k, n_jobs=-1)
			_path = global_path + 'per_' + str(k) + c +'.plk'
			if path.isfile(_path):
				per_trained = pickle.load(open(_path, 'rb'))
			else:
				print('Training the model...')
				per_trained = per.fit(X_train, y_train.values.ravel())
				with open(_path, 'wb') as handle:
					pickle.dump(per_trained, handle)
			pred = per_trained.predict(X_val)
			acc = metrics.accuracy_score(y_val, pred)
			met = metrics.classification_report(y_val, pred)
			all_accuracies.append(acc)
			if acc > all_accuracies[best_index]:
				best_index = i
				best_penalty = c
			repport.write('Perceptron with val and alpha = ' + str(k) +' penalty '+str(c) + ' :' + '\n')
			repport.write("Accuracy:" + str(acc) + '\n')
			repport.write('Full report: \n' + str(met) + '\n')
			del per_trained, per
			i += 1
	repport.write('In the test set with the best param '+ str(alphas[best_index])+' '+ best_penalty+'we have: \n')
	_path = global_path + 'per_' + str(alphas[best_index]) + best_penalty + '.plk'
	rf_trained = pickle.load(open(_path, 'rb'))
	pred = rf_trained.predict(X_test)
	acc = metrics.accuracy_score(y_test, pred)
	met = metrics.classification_report(y_test, pred)
	repport.write("Accuracy:" + str(acc) + '\n')
	repport.write('Full report: \n' + str(met) + '\n')
	repport.close()


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
	_path = global_path + 'perceptron.pkl'
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
	_path = global_path + 'mlp.pkl'
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
		random_forest(X_train, X_test, y_train, y_test, i)
	# knn(X_train, X_test, y_train, y_test, i)


def run_nn(X_train, X_test, X_val, y_train, y_test, y_val):
	reg = 0.1
	perceptron(X_train, X_test, X_val, y_train, y_test, y_val, 0.1, 'l2')
	mlp(X_train, X_test, X_val, y_train, y_test, y_val, 0.1)


def run_all():
	split_range = 0.2
	print('El split range es: ', split_range)
	(X_train, X_test, X_val, y_train, y_test, y_val) = split(split_range)
	naive_bayes(X_train, X_test, y_train, y_test)
	lda(X_train, X_test, y_train, y_test)
	qda(X_train, X_test, y_train, y_test)
	logistic_regresion(X_train, X_test, y_train, y_test)
	run_rf_knn(X_train, X_test, X_val, y_train, y_test, y_val)
	run_nn(X_train, X_test, X_val, y_train, y_test, y_val)


def main():
	print('El split range es: ', split_range)
	(X_train, X_test, X_val, y_train, y_test, y_val) = split(split_range)
	naive_bayes(X_train, X_test, y_train, y_test)
	lda(X_train, X_test, y_train, y_test)
	qda(X_train, X_test, y_train, y_test)
	logistic_regresion(X_train, X_test, y_train, y_test)
	run_rf_knn(X_train, X_test, X_val, y_train, y_test, y_val)


if __name__ == "__main__":
	main()
