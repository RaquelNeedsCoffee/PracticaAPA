import pandas as pd
import numpy as np
from mca import diagsvd
import functools
import sklearn.model_selection as ms
from sklearn.preprocessing import StandardScaler as stdc
import gc
import _pickle as pickle
categorical = ['source_system_tab','source_type','gender','city']
numerical = ['song_length', 'song_year']

def Matrix_mult(*args):
	"""An internal method to multiply matrices."""
	return functools.reduce(np.dot, args)


def my_mca(X_train):
	size = [20000, X_train.shape[1]]
	N_all = np.sum(X_train.values[0:size[0],:])

	Z_residual = X_train.values[0:size[0],:] / N_all
	# print('zshape', Z.shape)
	Sum_r = np.sum(Z_residual, axis=1)
	Sum_c = np.sum(Z_residual, axis=0)
	# print('sumc', Sum_c)
	print(size, Sum_r.shape, Sum_c.shape, N_all)
	# Compute residual
	Z_expected = np.outer(Sum_r, Sum_c)
	Z_residual -= Z_expected
	del Z_expected
	gc.collect()
	print('Z res')
	# Scale residual by the square root of column and row sums.
	# Note we are computing SVD on residual matrix, not the analogous covariance matrix,
	# Therefore, we are dividing by square root of Sums.
	# D_r = np.diag(Sum_r)
	# D_c = np.diag(Sum_c)
	D_r_sqrt_mi = np.sqrt(np.diag(Sum_r ** -1))
	del Sum_r
	gc.collect()
	D_c_sqrt_mi = np.sqrt(np.diag(Sum_c ** -1))
	del Sum_c
	gc.collect()
	print('end DC Dr')
	# print(Z_residual.shape, Z.shape, D_r_sqrt_mi.shape, D_c_sqrt_mi.shape)
	MCA_mat = Matrix_mult(D_r_sqrt_mi, Z_residual, D_c_sqrt_mi)
	## Apply SVD.
	## IN np implementation, MCA_mat = P*S*Q, not P*S*Q'
	P, S, Q = np.linalg.svd(MCA_mat)
	print(P.shape, S.shape, Q.shape)
	# Verify if MCA_mat = P*S*Q,
	S_d = diagsvd(S, size[0], size[1])
	# sum_mca = np.sum((Matrix_mult(P, S_d, Q) - MCA_mat) ** 2)
	# print('Difference between SVD and the MCA matrix is %0.2f' % sum_mca)
	# Compute factor space, or row and column eigen space
	F = Matrix_mult(D_r_sqrt_mi, P, S_d)  ## Column Space, contains linear combinations of columns
	G = Matrix_mult(D_c_sqrt_mi, Q.T, S_d.T)  ## Row space, contains linear combinations of rows
	pickle.dump(F, open('F.pkl','wb'))
	pickle.dump(G, open('G.pkl', 'wb'))
	pickle.dump(S, open('S.pkl', 'wb'))
	del  Q, P, MCA_mat, S_d, D_r_sqrt_mi, D_c_sqrt_mi
	gc.collect()
	print(F.shape, G.shape)
	Lam = S ** 2
	Expl_var = Lam / np.sum(Lam)

	print('Eigen values are ', Lam)
	print('Explained variance of eigen vectors are ', Expl_var)
	# K = 10
	# E = np.array([(K / (K - 1.) * (lm - 1. / K)) ** 2 if lm > 1. / K else 0 for lm in S ** 2])
	# Expl_var_bn = E / np.sum(E)
	# print('Eigen vectors after Benzécri correction are ', E)
	# print('Explained variance of eigen vectors after Benzécri correction are ', Expl_var_bn)
	# J = 22.
	# green_norm = (K / (K - 1.) * (np.sum(S ** 4) - (J - K) / K ** 2.))
	# # J is the number of categorical variables. 22 in our case.

	# print('Explained variance of eigen vectors after Greenacre correction are ', E / green_norm)
	data = {'Iλ': pd.Series(Lam),
		'τI': pd.Series(Expl_var),
		# 'Zλ': pd.Series(E),
		# 'τZ': pd.Series(Expl_var_bn),
		# 'cλ': pd.Series(E),
		# 'τc': pd.Series(E / green_norm),
	}
	columns = ['Iλ', 'τI']#, 'Zλ', 'τZ', 'cλ', 'τc']
	table2 = pd.DataFrame(data=data, columns=columns).fillna(0)
	table2.index += 1
	table2.loc['Σ'] = table2.sum()
	table2.index.name = 'Factor'
	np.round(table2.astype(float), 4)
	table2.to_csv('table2.csv')
	columns_keep = 10
	print('Varianza explicada tomando 5 features: ', table2['τI'][0:columns_keep].sum())
	## The projection can also be computed using vectorized form,

	X_train = Matrix_mult(X_train.values, G[:, :columns_keep]) / S[:columns_keep] / 10
	X_test = pd.read_csv('../Data/subset_cat_Test.csv', compact_ints=True)
	X_test = Matrix_mult(X_test.values, G[:, :columns_keep]) / S[:columns_keep] / 10
	X_val = pd.read_csv('../Data/subset_cat_Val.csv', compact_ints=True)
	X_val = Matrix_mult(X_val.values, G[:, :columns_keep]) / S[:columns_keep] / 10
	T_cat = pd.read_csv('../Data/subset_cat_T.csv')
	T_cat = Matrix_mult(T_cat.values, G[:, :columns_keep]) / S[:columns_keep] / 10
	print('shape', X_test.shape)
	names = ['V' + str(i) for i in range(columns_keep)]
	return pd.DataFrame(X_train, columns=names), pd.DataFrame(X_test, columns=names), pd.DataFrame(X_val, columns=names),  pd.DataFrame(T_cat, columns=names)


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
	      X_val.shape, '\n Type: ',type(X_train))

	return X_train, X_test, X_val, y_train, y_test, y_val


def standarize_data(df):
	df[numerical] = stdc().fit_transform(df[numerical])
	return df


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
	T = pd.read_csv('../Data/def_test.csv')
	T_num = T[numerical]

	X_num = X[numerical]
	X_num = standarize_data(X_num)
	X['city'] = X['city'].astype('category')
	T['city'] = T['city'].astype('category')
	DummiesX = pd.get_dummies(data=X[categorical], prefix_sep='|')
	T_cat = pd.get_dummies(data=T[categorical], prefix_sep='|')
	y = X['target']
	print('dummies size: ', DummiesX.shape)
	(X_num_train, X_num_test, X_num_val, y_train, y_test, y_val) = split(X_num, y, 0.5)
	del X_num, X
	gc.collect()
	X_num_train.to_csv('../Data/subset_num_Train.csv')
	X_num_test.to_csv('../Data/subset_num_Test.csv')
	X_num_val.to_csv('../Data/subset_num_Val.csv')
	T_num.to_csv('../Data/subset_num_T.csv')
	y_train.to_csv('../Data/preprocessed_y_Train.csv')
	y_test.to_csv('../Data/preprocessed_y_Test.csv')
	y_val.to_csv('../Data/preprocessed_y_Val.csv')

	(X_cat_train, X_cat_test, X_cat_val, y_train, y_test, y_val) = split(DummiesX, y, 0.5)
	del DummiesX
	gc.collect()
	print('end split')
	X_cat_train[X_cat_train.columns] = X_cat_train[X_cat_train.columns].astype(np.uint8)
	X_cat_test[X_cat_test.columns] = X_cat_test[X_cat_test.columns].astype(np.uint8)
	X_cat_val[X_cat_val.columns] = X_cat_val[X_cat_val.columns].astype(np.uint8)
	T_cat[T_cat.columns] = T_cat[T_cat.columns].astype(np.uint8)
	T_cat.to_csv('../Data/T_cat.csv')
	X_cat_train.to_csv('../Data/X_cat_Train.csv')
	X_cat_test.to_csv('../Data/X_cat_Test.csv')
	X_cat_val.to_csv('../Data/X_cat_Val.csv')
	print('loading X train... ')
	X_cat_train = pd.read_csv('../Data/X_cat_Train.csv', compact_ints=True)
	print('loaded')

	T_cat = pd.read_csv('../Data/T_cat.csv', compact_ints=True)


	T_sum =  (T_cat.values.sum(axis=0) != 0)
	X_train_sum = (X_cat_train.values.sum(axis=0) != 0)
	X_cat_test = pd.read_csv('../Data/X_cat_Test.csv', compact_ints=True)
	print('loaded')
	X_test_sum = (X_cat_test.values.sum(axis=0) != 0)
	print('x_test end')
	print('loading X val... ')
	X_cat_val = pd.read_csv('../Data/X_cat_Val.csv', compact_ints=True)
	print('loaded')
	X_val_sum = (X_cat_val.values.sum(axis=0) != 0)
	print('x val end')
	index =np.logical_and(np.logical_and(X_train_sum, np.logical_and(X_test_sum, X_val_sum)),T_sum)
	print('index end')
	X_cat_train = X_cat_train.loc[:, index]
	X_cat_test = X_cat_test.loc[:, index]
	X_cat_val = X_cat_val.loc[:, index]
	T_cat = T_cat.loc[:,index]
	gc.collect()
	print('loc end')
	T_cat.to_csv('../Data/subset_cat_T.csv')
	X_cat_train.to_csv('../Data/subset_cat_Train.csv')
	X_cat_test.to_csv('../Data/subset_cat_Test.csv')
	X_cat_val.to_csv('../Data/subset_cat_Val.csv')
	del X_cat_val,X_cat_train, X_cat_test, X_num_test, X_num_train, X_num_val, T_cat, T_num, T_sum
	gc.collect()

def preprocess():

	X_cat_train = pd.read_csv('../Data/subset_cat_Train.csv', compact_ints=True)
	# X_cat_train = pd.read_csv('../Data/subset_cat_TrainMCA.csv', compact_ints=True)
	# X_cat_test = pd.read_csv('../Data/subset_cat_TestMCA.csv', compact_ints=True)
	# X_cat_val = pd.read_csv('../Data/subset_cat_ValMCA.csv', compact_ints=True)
	# T_cat = pd.read_csv('../Data/subset_cat_TMCA.csv', compact_ints=True)
	# print('end csv')
	X_cat_train, X_cat_test, X_cat_val, T_cat = my_mca(X_cat_train)
	X_cat_train.to_csv('../Data/subset_cat_TrainMCA.csv')
	X_cat_test.to_csv('../Data/subset_cat_TestMCA.csv')
	X_cat_val.to_csv('../Data/subset_cat_ValMCA.csv')
	T_cat.to_csv('../Data/subset_cat_TMCA.csv')


	X_num_train = pd.read_csv('../Data/subset_num_Train.csv')
	X_num_test = pd.read_csv('../Data/subset_num_Test.csv')
	X_num_val = pd.read_csv('../Data/subset_num_Val.csv')
	T_num = pd.read_csv('../Data/subset_num_T.csv')

	concat_Train = pd.concat([X_cat_train, X_num_train], axis=1, ignore_index=True)
	concat_Test = pd.concat([X_cat_test, X_num_test], axis=1, ignore_index=True)
	concat_Val = pd.concat([X_cat_val, X_num_val], axis=1, ignore_index=True)
	concat_T= pd.concat([T_cat, T_num], axis=1, ignore_index=True)

	y_train = pd.read_csv('../Data/preprocessed_y_Train.csv')
	y_test = pd.read_csv('../Data/preprocessed_y_Test.csv')
	y_val = pd.read_csv('../Data/preprocessed_y_Val.csv')

	concat_Train = pd.concat([concat_Train, y_train], axis=1, ignore_index=True)
	concat_Test = pd.concat([concat_Test, y_test], axis=1, ignore_index=True)
	concat_Val = pd.concat([concat_Val, y_val], axis=1, ignore_index=True)

	# concat_Train = concat_Train.rename(columns ={9:'target'})
	# concat_Test = concat_Test.rename(columns ={9:'target'})
	# concat_Val = concat_Val.rename(columns ={9:'target'})

	print('concat X Train ', concat_Train.shape)
	print('concat X ', concat_Test.shape)
	print('concat X ', concat_Val.shape)
	print('concat T ', concat_T.shape)
	concat_T.to_csv('../Data/preprocessedOriginalTMCA.csv')
	concat_Train.to_csv('../Data/preprocessedTrainMCA.csv')
	concat_Test.to_csv('../Data/preprocessedTestMCA.csv')
	concat_Val.to_csv('../Data/preprocessedValMCA.csv')



def data_from_files():
	X_train = pd.read_csv('../Data/preprocessedTrain.csv', compact_ints=True)
	X_test = pd.read_csv('../Data/preprocessedTest.csv', compact_ints=True)
	X_val = pd.read_csv('../Data/preprocessedVal.csv', compact_ints=True)

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

def data_from_filesMCA():
	X_train = pd.read_csv('../Data/preprocessedTrainMCA.csv', compact_ints=True)
	X_test = pd.read_csv('../Data/preprocessedTestMCA.csv', compact_ints=True)
	X_val = pd.read_csv('../Data/preprocessedValMCA.csv', compact_ints=True)

	X_train = X_train.rename(columns={'9': 'target'})
	X_test = X_test.rename(columns={'9': 'target'})
	X_val = X_val.rename(columns={'9': 'target'})

	y_train = X_train['target']
	y_test = X_test['target']
	y_val = X_val['target']
	print('\nLoaded data:')
	X_train = X_train.drop(columns=['target','Unnamed: 0'])
	X_test = X_test.drop(columns=['target','Unnamed: 0'])
	X_val = X_val.drop(columns=['target','Unnamed: 0'])
	print('Train shape: ', X_train.shape)
	print('Train shape Y: ', y_train.shape)
	print('Test shape: ', X_test.shape)
	print('Test shape Y: ', y_test.shape)
	print('Val shape: ', X_val.shape)
	print('Test shape: ', y_val.shape)
	return stdc().fit_transform(X_train), stdc().fit_transform(X_test), stdc().fit_transform(X_val), y_train, y_test, y_val

def main():
	file = 'def_training.csv'
	X = pd.read_csv('../Data/' + file)
	# generate_partition(X)
	preprocess()
	# data_from_files()


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
