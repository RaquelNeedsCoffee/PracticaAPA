import pandas as pd
import numpy as np
from mca import diagsvd
import functools
import sklearn.model_selection as ms
from sklearn.preprocessing import StandardScaler as stdc
import gc
categorical = ['source_system_tab', 'source_screen_name', 'source_type',
	'gender', 'genre_ids', 'language', 'country_code',
	]  # 'artist_name', 'composer' , 'lyricist', 'age_range']'registrant_code' 'city', 'registered_via',
numerical = ['registration_init_time', 'song_length', 'registration_init_time', 'expiration_date', 'song_year']


def Matrix_mult(*args):
	"""An internal method to multiply matrices."""
	return functools.reduce(np.dot, args)


def my_mca(X_train, X_test, X_val):
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
	X_val = Matrix_mult(X_val.values, G[:, :45]) / S[:45] / 10
	print('shape', X_test.shape)
	names = ['V' + str(i) for i in range(45)]
	return pd.DataFrame(X_train, columns=names), pd.DataFrame(X_test, columns=names), pd.DataFrame(X_val, columns=names)


def to_dummies(X):
	for column in X.columns:
		X[column] = X[column].astype('category')
	return pd.get_dummies(X)


def split(X, y, proportion, name):
	print('Train \nRaw data: \n ', 'Shape: ', X.shape, 'Type: ', type(X))
	print('Target \nRaw data: \n ', 'Shape: ', y.shape)
	# I split the data between the X and the target value

	# X is a dataframe not a np.array
	print('Size of X: ', X.shape, '\nSize of y: ', y.shape)
	(X_train, X_test, y_train, y_test) = ms.train_test_split(X, y, test_size=proportion, random_state=1, stratify=y)
	(X_test, X_val, y_test, y_val) = ms.train_test_split(X_test, y_test, test_size=.5, random_state=1, stratify=y_test)
	print('\n New train shape: ', X_train.shape, ' \n New test shape: ', X_test.shape, '\n New val shape: ',
	      X_val.shape, '\n Type: ',type(X_train))
	# np.savez('../Data/X_train_'+name+'_matrix.npz',X_train )
	# np.savez('../Data/X_test_'+name+'_matrix.npz', X_test)
	# np.savez('../Data/X_val_'+name+'_matrix.npz',X_val)
	# np.savez('../Data/y_train_' + name + '_matrix.npz', y_train)
	# np.savez('../Data/y_test_' + name + '_matrix.npz', y_test)
	# np.savez('../Data/y_val_' + name + '_matrix.npz', y_val)
	# if name == 'cat':
	# 	np.savez('../Data/X_train_' + name + '_columns.npz', X_train.columns)
	# 	np.savez('../Data/X_test_' + name + '_columns.npz', X_test.columns)
	# 	np.savez('../Data/X_val_' + name + '_columns.npz', X_val.columns)

	return X_train, X_test, X_val, y_train, y_test, y_val


def standarize_data(df):
	df[numerical] = stdc().fit_transform(df[numerical])
	return df


def remove_minor_categories(X):
	# todo: borrar las categorías que salen menos
	pass

def clean_array():
	name = 'cat'
	X_cat_train=np.load('../Data/X_train_' + name + '_matrix.npz')['arr_0']
	X_cat_test = np.load('../Data/X_test_' + name + '_matrix.npz')['arr_0']
	X_cat_val = np.load('../Data/X_val_' + name + '_matrix.npz')['arr_0']
	X_cat_train_columns = np.load('../Data/X_train_' + name + '_matrix_columns.npz')['arr_0']
	X_cat_test_columns = np.load('../Data/X_test_' + name + '_matrix_columns.npz')['arr_0']
	X_cat_val_columns = np.load('../Data/X_val_' + name + '_matrix_columns.npz')['arr_0']
	X_train_sum = (X_cat_train.sum(axis=0) != 0)
	print('xtrain sum end')
	X_test_sum = (X_cat_test.sum(axis=0) != 0)
	print('x_test end')
	X_val_sum = (X_cat_val.sum(axis=0) != 0)
	print('x val end')
	index = np.logical_and(X_train_sum, np.logical_and(X_test_sum, X_val_sum))
	print('index end')
	X_cat_train = X_cat_train[:, index]
	X_cat_test = X_cat_test[:, index]
	X_cat_val = X_cat_val[:, index]
	np.savez('../Data/X_train_' + name + '_submatrix.npz', X_cat_train)
	np.savez('../Data/X_test_' + name + '_submatrix.npz', X_cat_test)
	np.savez('../Data/X_val_' + name + '_submatrix.npz', X_cat_val)

	X_train = pd.SparseDataFrame(data=X_cat_train, columns=X_cat_train_columns)
	X_test = pd.SparseDataFrame(data=X_cat_test, columns=X_cat_test_columns)
	X_val = pd.SparseDataFrame(data=X_cat_val, columns=X_cat_val_columns)
	X_train.to_csv('../Data/subset_cat_Train.csv')
	X_test.to_csv('../Data/subset_cat_Test.csv')
	X_val.to_csv('../Data/subset_cat_Val.csv')
	return X_train,X_test,X_val

def generate_partition(X):
	X_num = X[numerical]
	X_num = standarize_data(X_num)
	DummiesX = pd.get_dummies(data=X[categorical], columns=categorical, prefix_sep='|')
	y = X['target']
	print('dummies size: ', DummiesX.shape)
	(X_num_train, X_num_test, X_num_val, y_train, y_test, y_val) = split(X_num, y, 0.4, 'num')

	X_num_train.to_csv('../Data/subset_num_Train.csv')
	X_num_test.to_csv('../Data/subset_num_Test.csv')
	X_num_val.to_csv('../Data/subset_num_Val.csv')
	y_train.to_csv('../Data/preprocessed_y_Train.csv')
	y_test.to_csv('../Data/preprocessed_y_Test.csv')
	y_test.to_csv('../Data/preprocessed_y_Val.csv')

	(X_cat_train, X_cat_test, X_cat_val, y_train, y_test, y_val) = split(DummiesX, y, 0.4, 'cat')
	del X, DummiesX
	gc.collect()
	print('end split')
	# X_cat_train, X_cat_test, X_cat_val = clean_array()

	X_cat_train.to_csv('../Data/X_cat_Train.csv')
	X_cat_test.to_csv('../Data/X_cat_Test.csv')
	X_cat_val.to_csv('../Data/X_cat_Val.csv')

def preprocess(X):
	print('loading... ')
	X_cat_train = pd.read_csv('../Data/X_cat_Train.csv')


	print('loaded')

	X_train_sum = (X_cat_train.values.sum(axis=0) != 0)
	print('xtrain sum end')
	X_cat_test = pd.read_csv('../Data/X_cat_Test.csv')
	X_test_sum = (X_cat_test.values.sum(axis=0) != 0)
	print('x_test end')
	X_cat_val = pd.read_csv('../Data/X_cat_Val.csv')
	X_val_sum = (X_cat_val.values.sum(axis=0) != 0)
	print('x val end')
	index = np.logical_and(X_train_sum,np.logical_and(X_test_sum,X_val_sum))
	print('index end')
	X_cat_train = X_cat_train.loc[:,index]
	X_cat_test = X_cat_test.loc[:, index]
	X_cat_val = X_cat_val.loc[:,index]
	gc.collect()
	print('loc end')
	X_cat_train.to_csv('../Data/subset_cat_Train.csv')
	X_cat_test.to_csv('../Data/subset_cat_Test.csv')
	X_cat_val.to_csv('../Data/subset_cat_Val.csv')

	print('end csv')
	# for feature in X_cat_train.columns:
	# 	if np.sum(X_cat_train[feature]) == 0 or np.sum(X_cat_test[feature]) == 0 or np.sum(X_cat_val[feature]) == 0:
	# 		# print(feature)
	# 		X_cat_train.drop(feature, axis=1)
	# 		X_cat_test.drop(feature, axis=1)
	# 		X_cat_val.drop(feature, axis=1)
	print('droped columns')
	X_cat_train, X_cat_test, X_cat_val = my_mca(X_cat_train, X_cat_test, X_cat_val)

	X_num_train = pd.read_csv('../Data/subset_num_Train.csv')
	X_num_test = pd.read_csv('../Data/subset_num_Test.csv')
	X_num_val = pd.read_csv('../Data/subset_num_Val.csv')

	concat_Train = pd.concat([X_cat_train, X_num_train], axis=1, ignore_index=True)
	concat_Test = pd.concat([X_cat_test, X_num_test], axis=1, ignore_index=True)
	concat_Val = pd.concat([X_cat_val, X_num_val], axis=1, ignore_index=True)

	print('concat X Train ', concat_Train.shape)
	print('concat X ', concat_Test.shape)
	print('concat X ', concat_Val.shape)
	concat_Train.to_csv('../Data/preprocessedTrain.csv')
	concat_Test.to_csv('../Data/preprocessedTest.csv')
	concat_Val.to_csv('../Data/preprocessedVal.csv')



def data_from_files():
	X_train = pd.read_csv('../Data/preprocessedTrain.csv')
	X_test = pd.read_csv('../Data/preprocessedTest.csv')
	X_val = pd.read_csv('../Data/preprocessedVal.csv')

	y_train = pd.read_csv('../Data/preprocessed_y_Train.csv')
	y_test = pd.read_csv('../Data/preprocessed_y_Test.csv')
	y_val = pd.read_csv('../Data/preprocessed_y_Val.csv')

	print('\nLoaded data:')
	print('Train shape: ', X_train.shape)
	print('Test shape: ', X_test.shape)
	print('Val shape: ', X_val.shape)
	return X_train, X_test, X_val, y_train, y_test, y_val

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
