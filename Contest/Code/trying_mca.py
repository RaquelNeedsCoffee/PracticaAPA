import pandas as pd
import numpy as np
from mca import diagsvd
import functools
import sklearn.model_selection as ms
from sklearn.preprocessing import StandardScaler as stdc

categorical = ['source_system_tab', 'source_screen_name', 'source_type', 'city', 'registered_via',
               'gender', 'genre_ids', 'artist_name', 'composer', 'lyricist', 'language', 'country_code',
               'registrant_code']
numerical = ['registration_init_time', 'song_length', 'registration_init_time', 'expiration_date', 'song_year']


def Matrix_mult(*args):
	"""An internal method to multiply matrices."""
	return functools.reduce(np.dot, args)


def my_mca(X_train, X_test):
	X_values = X_train.values
	N_all = np.sum(X_values)
	Z = X_values / N_all
	# print('zshape', Z.shape)
	Sum_r = np.sum(Z, axis=1)
	Sum_c = np.sum(Z, axis=0)
	# print('sumc', Sum_c)
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
	# Todo: calcular cuantas features necesitamos para conservar x varianza
	print('Varianza explicada tomando 45 features: ', table2['τZ'][0:45].sum())
	## The projection can also be computed using vectorized form,
	X_train = Matrix_mult(X_train.values, G[:, :45]) / S[:45] / 10
	X_test = Matrix_mult(X_test.values, G[:, :45]) / S[:45] / 10
	print('shape', X_test.shape)
	names = ['V' + str(i) for i in range(45)]
	return pd.DataFrame(X_train, columns=names), pd.DataFrame(X_test, columns=names)


def to_dummies(X):
	for column in X.columns:
		X[column] = X[column].astype('category')
	return pd.get_dummies(X)


def split(X, y, proportion):
	print('Train \nRaw data: \n ', 'Shape: ', X.shape, 'Type: ', type(X))
	print('Target \nRaw data: \n ', 'Shape: ', y.shape)
	# I split the data between the X and the target value

	# X is a dataframe not a np.array
	print('Size of X: ', X.shape, '\nSize of y: ', y.shape)
	(X_train, X_test, y_train, y_test) = ms.train_test_split(X, y, test_size=proportion, random_state=1, stratify=y)
	(X_test, X_val, y_test, y_val) = ms.train_test_split(X_test, y_test, test_size=.5, random_state=1, stratify=y_test)
	print('\n New train shape: ', X_train.shape, ' \n New test shape: ', X_test.shape, '\n New val shape: ',
	      X_val.shape)
	X_train = pd.DataFrame(data=X_train.values, columns=X_train.columns)
	X_test = pd.DataFrame(data=X_test.values, columns=X_test.columns)
	return X_train, X_test, X_val, y_train, y_test, y_val


def standarize_data(df):
	df[numerical] = stdc().fit_transform(df[numerical])
	return df

def remove_minor_categories():
	pass

def preprocess(X):
	# todo: si no revienta probamso con el name
	# X_cat = X[categorical]
	# X_num = X[numerical]
	# X_num = standarize_data(X_num)
	categorical = ['source_system_tab', 'source_screen_name', 'source_type', 'city', 'registered_via',
	               'gender', 'genre_ids',  'country_code',
	               'registrant_code']

	DummiesX = pd.get_dummies(data=X,columns=categorical, prefix_sep='|', sparse=True)
	# y = X['target']
	# print('dummies size: ', DummiesX.shape)
	# print('dummies nas:', np.sum(DummiesX.isnull().sum()))
	# (X_cat_train, X_cat_test, X_cat_val, y_train, y_test, y_val) = split(DummiesX, y, 0.3)
	# (X_num_train, X_num_test, X_num_val, y_train, y_test, y_val) = split(X_num, y, 0.3)
	# for feature in X_cat_train.columns:
	# 	if np.sum(X_cat_train[feature]) == 0 or np.sum(X_cat_test[feature]) == 0 or np.sum(X_cat_val[feature]) == 0:
	# 		# print(feature)
	# 		X_cat_train.drop(feature, axis=1)
	# 		X_cat_test.drop(feature, axis=1)
	# 		X_cat_val.drop(feature, axis=1)
	# for i in range(len(X_cat_train.values)):
	# 	if np.sum(X_cat_train.ix[i, :]) == 0:
	# 		print('PETA')
	# print('X_cat nas:', np.sum(X_cat_train.isnull().sum()))
	# my_mca(X_cat_train, X_cat_test)
	# concatX = pd.concat([DummiesX, X_num], axis=1, ignore_index=True)
	# print('concat X ', concatX.shape)
	# concatX.to_csv('bicho.csv')


def main():
	file = 'def_training.csv'
	X = pd.read_csv('../Data/' + file)
	preprocess(X)


# (X_train, X_test, X_val, y_train, y_test, y_val) = split(0.3, 'samples/definitivo.csv')
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
