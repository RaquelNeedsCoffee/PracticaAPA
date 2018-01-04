import pandas as pd
import numpy as np
from os import path
import warnings
import _pickle as pickle

global_path = '../Data/Models/'

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


def data_from_filesMCA():
	print('loading data...')
	warnings.filterwarnings('ignore')
	X_train = pd.read_csv('../Data/preprocessedTrainMCA.csv', compact_ints=True)
	X_test = pd.read_csv('../Data/preprocessedTestMCA.csv', compact_ints=True)
	X_val = pd.read_csv('../Data/preprocessedValMCA.csv', compact_ints=True)

	y_train = X_train['target']
	y_test = X_test['target']
	y_val = X_val['target']
	print('\nLoaded data:')
	X_train = X_train.drop(columns=['target'])
	X_test = X_test.drop(columns=['target'])
	X_val = X_val.drop(columns=['target'])
	print('Train shape: ', X_train.shape)
	print('Train shape Y: ', y_train.shape)
	print('Test shape: ', X_test.shape)
	print('Test shape Y: ', y_test.shape)
	print('Val shape: ', X_val.shape)
	print('Test shape: ', y_val.shape)
	return X_train, X_test, X_val, y_train, y_test, y_val


def save_model(name, model):
	with open(global_path + name, 'wb') as handle:
		pickle.dump(model, handle)


def load_model(name):
	if path.isfile(global_path + name):
		model = pickle.load(open(global_path + name, 'rb'))
		return model
	else:
		print("There's no model in this path")
		return None
