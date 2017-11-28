
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import gc
from sklearn.preprocessing import Imputer

def show_missing(data):
    missing = data.columns[data.isnull().any()].tolist()
    return missing


def print_missing(data):
    print(data[show_missing(data)].isnull().sum())


def imputting_values(data):
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imp.fit(data)
    imp.transform(data)
    return data


def load_all():
    train = pd.read_csv('../Data/train.csv')
    test = pd.read_csv('../Data/test.csv')
    members = pd.read_csv('../Data/members.csv')
    songs = pd.read_csv('../Data/songs.csv')

    train_merged = train.merge(songs, how='left', on='song_id').merge(members, how='left', on='msno')

    test_merged = test.merge(songs, how='left', on='song_id').merge(members, how='left', on='msno')
    del members, songs
    gc.collect()
    print_missing(train_merged)
    train_merged = imputting_values(train_merged)
    print_missing(train_merged)





def main():
    load_all()


if __name__ == "__main__":
    main()