import pandas as pd
import sklearn.model_selection as ms
from sklearn import preprocessing


def split(test_proportion, file='def_training.csv'):
	"""
	This function does the stratified split of our data with a test size of 0.95 (as the teacher recommended )
	We will have a train file with the 5% of the data and the val and test with the rest
	:return: X_train, X_test, y_train, y_test
	"""
	contest_data = pd.read_csv('../Data/' + file, header=0)

	X = contest_data.ix[:, contest_data.columns != 'target']
	y = contest_data.ix[:, contest_data.columns == 'target']
	print('Train \nRaw data: \n ', 'Shape: ', X.shape, 'Type: ', type(X))
	print('Target \nRaw data: \n ', 'Shape: ', y.shape)
	# I split the data between the X and the target value

	# X is a dataframe not a np.array
	print('Size of X: ', X.shape, '\nSize of y: ', y.shape)
	(X_train, X_test, y_train, y_test) = ms.train_test_split(X, y, test_size=test_proportion, random_state=1,
	                                                         stratify=y)
	(X_test, X_val, y_test, y_val) = ms.train_test_split(X_test, y_test, test_size=.5, random_state=1, stratify=y_test)
	print('\n New train shape: ', X_train.shape, ' \n New test shape: ', X_test.shape, '\n New val shape: ',
	      X_val.shape)
	return X_train, X_test, X_val, y_train, y_test, y_val


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
	(X_train, X_test, X_val, y_train, y_test, y_val) = split(0.95)
	print(X_train.head())


if __name__ == "__main__":
	main()
