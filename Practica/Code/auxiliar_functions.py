import pandas as pd
import numpy as np
from os import path
import warnings
import _pickle as pickle
import operator
import math

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


def data_from_files():
    print('loading data...')
    warnings.filterwarnings('ignore')
    X_train = pd.read_csv('../Data/clean_train.csv', compact_ints=True)
    X_test = pd.read_csv('../Data/clean_test.csv', compact_ints=True)
   
    y_train = X_train['target']
    y_test = X_test['target']
    print('\nLoaded data:')
    X_train = X_train.drop(columns=['target'])
    X_test = X_test.drop(columns=['target'])
    print('Train shape: ', X_train.shape)
    print('Train shape Y: ', y_train.shape)
    print('Test shape: ', X_test.shape)
    print('Test shape Y: ', y_test.shape)
   
    return X_train, y_train, X_test,y_test,


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


def extract_info(df):
    n_rows = len(df)
    nulls_sum = df.isnull().sum().rename('null_count')
    nulls_per100 = nulls_sum.apply(lambda x: (100*x)/n_rows)
    nulls_per100 = nulls_per100.rename('null_percentage')
    df_types = df.dtypes.rename('dtypes')
    info_series = pd.concat([nulls_sum, nulls_per100, df_types], axis=1)
    return info_series


def print_df_info(df):
    info_series = extract_info(df)
    n_rows = len(df)
    print(info_series, '\n')
    print('nrows: {}\n'.format(n_rows))
    print('Memory consumed by dataframe : {} MB\n'.format(df.memory_usage(index=True).sum() / 1024 ** 2))
