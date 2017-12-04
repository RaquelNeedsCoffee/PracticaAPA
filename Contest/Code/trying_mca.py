import pandas as pd
import numpy as np
from mca import *
from time import time
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn import preprocessing
from Contest.Code.split_data import split
import functools


def Matrix_mult(*args):
	"""An internal method to multiply matrices."""
	return functools.reduce(np.dot, args)


def my_mca(X_train, X_test):
	X_values = X_train.values
	N_all = np.sum(X_values)
	Z = X_values / N_all
	Sum_r = np.sum(Z, axis=1)
	Sum_c = np.sum(Z, axis=0)
	print(X_values.shape, Sum_r.shape, Sum_c.shape, N_all)
	# Compute residual
	Z_expected = np.outer(Sum_r, Sum_c)
	Z_residual = Z - Z_expected

	# Scale residual by the square root of column and row sums.
	# Note we are computing SVD on residual matrix, not the analogous covariance matrix,
	# Therefore, we are dividing by square root of Sums.
	D_r = np.diag(Sum_r)
	D_c = np.diag(Sum_c)

	D_r_sqrt_mi = np.sqrt(np.diag(Sum_r ** -1))
	D_c_sqrt_mi = np.sqrt(np.diag(Sum_c ** -1))

	print(Z_residual.shape, Z.shape, D_r_sqrt_mi.shape, D_c_sqrt_mi.shape)
	MCA_mat = Matrix_mult(D_r_sqrt_mi, Z_residual, D_c_sqrt_mi)
	## Apply SVD.
	## IN np implementation, MCA_mat = P*S*Q, not P*S*Q'
	P, S, Q = np.linalg.svd(MCA_mat)
	print(P.shape, S.shape, Q.shape)
	# Verify if MCA_mat = P*S*Q,
	S_d = diagsvd(S, X_values.shape[0], X_values.shape[1])
	sum_mca = np.sum((Matrix_mult(P, S_d, Q) - MCA_mat) ** 2)
	print('Difference between SVD and the MCA matrix is %0.2f' % sum_mca)
	# Compute factor space, or row and column eigen space
	F = Matrix_mult(D_r_sqrt_mi, P, S_d)  ## Column Space, contains linear combinations of columns
	G = Matrix_mult(D_c_sqrt_mi, Q.T, S_d.T)  ## Row space, contains linear combinations of rows

	print(F.shape, G.shape)
	Lam = S ** 2
	Expl_var = Lam / np.sum(Lam)

	print('Eigen values are ', Lam)
	print('Explained variance of eigen vectors are ', Expl_var)
	K = 10
	E = np.array([(K / (K - 1.) * (lm - 1. / K)) ** 2 if lm > 1. / K else 0 for lm in S ** 2])
	Expl_var_bn = E / np.sum(E)
	print('Eigen vectors after Benzécri correction are ', E)
	print('Explained variance of eigen vectors after Benzécri correction are ', Expl_var_bn)
	J = 22.
	green_norm = (K / (K - 1.) * (np.sum(S ** 4) - (J - K) / K ** 2.))
	# J is the number of categorical variables. 22 in our case.

	print('Explained variance of eigen vectors after Greenacre correction are ', E / green_norm)
	data = {'Iλ': pd.Series(Lam),
	        'τI': pd.Series(Expl_var),
	        'Zλ': pd.Series(E),
	        'τZ': pd.Series(Expl_var_bn),
	        'cλ': pd.Series(E),
	        'τc': pd.Series(E / green_norm),
	        }
	columns = ['Iλ', 'τI', 'Zλ', 'τZ', 'cλ', 'τc']
	table2 = pd.DataFrame(data=data, columns=columns).fillna(0)
	table2.index += 1
	table2.loc['Σ'] = table2.sum()
	table2.index.name = 'Factor'
	np.round(table2.astype(float), 4)
	#Todo: calcular cuantas features necesitamos para conservar x varianza
	print('Varianza explicada tomando 45 features: ', table2['τZ'][0:45].sum())
	## The projection can also be computed using vectorized form,
	X_train = Matrix_mult(X_train.values, G[:, :45]) / S[:45] / 10
	X_test = Matrix_mult(X_test.values, G[:, :45]) / S[:45] / 10
	print('shape',X_test.shape)
	names = ['V' +str(i) for i in range(45)]
	return pd.DataFrame(X_train, columns=names), pd.DataFrame(X_test, columns=names)


def to_dummies(X):
	X['language'] = X['language'].astype('category')
	return pd.get_dummies(X)

def main():
	categorical = ['source_system_tab', 'source_screen_name', 'source_type', 'city', 'registered_via',
	               'gender', 'genre_ids', 'artist_name', 'composer', 'lyricist', 'language', 'name', 'country_code', 'registrant_code']
	numerical = ['registration_init_time', 'song_length', 'registration_init_time', 'expiration_date', 'song_year']
	file = 'samples/definitivo.csv'
	X = pd.read_csv('../Data/' + file, header=0)
	X_cat = X[categorical]
	X_num = X[numerical]
	DummiesX = to_dummies(X_cat)
	DummiesX.to_csv('dummies.csv')
	(X_train, X_test, X_val, y_train, y_test, y_val) = split(0.3, 'samples/definitivo.csv')
	#
	# new_X_train_cat, new_X_test_cat = my_mca(dumy_train, dumy_test)
	# print(new_X_test_cat.shape)
	#
	# concaTrain= pd.concat([X_train_num, new_X_train_cat], axis=1, ignore_index=True)
	# # concaTest = pd.concat([X_test_num, new_X_test_cat], axis=1, ignore_index=True)
	#
	# print(concaTrain.shape)
	# print(concaTest.shape)



if __name__ == "__main__":
	main()
