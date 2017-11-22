import pandas as pd
import sklearn.model_selection as ms

from Contest.Code.optimice_dataset import optimice

def split(test_proportion):
    """
    This function does the stratified split of our data with a test size of 0.95 (as the teacher recommended )
    We will have a train file with the 5% of the data and the val and test with the rest
    :return: X_train, X_test, y_train, y_test
    """
    contest_train = pd.read_csv('../Data/clean_train.csv', header=0)
    contest_target = pd.read_csv('../Data/clean_target.csv', header=0)
    print('Train \nRaw data: \n ', 'Shape: ', contest_train.shape, 'Type: ', type(contest_target))
    print('Target \nRaw data: \n ', 'Shape: ', contest_target.shape)
    # I split the data between the X and the target value
    X = contest_train
    y = contest_target.ix[:,1]

    # X is a dataframe not a np.array
    print('Size of X: ', X.shape, '\nSize of y: ', y.shape)
    (X_train, X_test, y_train, y_test) = ms.train_test_split(X, y, test_size=test_proportion, random_state=1, stratify=y)
    (X_test, X_val, y_test, y_val) = ms.train_test_split(X_test, y_test, test_size=.5, random_state=1, stratify=y_test)
    print('\n New train shape: ', X_train.shape, ' \n New test shape: ', X_test.shape, '\n New val shape: ', X_val.shape)
    return optimice(X_train), optimice(X_test), optimice(X_val), optimice(y_train), optimice(y_test), optimice(y_val)


def main():
    (X_train, X_test, X_val, y_train, y_test,y_val) = split(0.95)


if __name__ == "__main__":
    main()

