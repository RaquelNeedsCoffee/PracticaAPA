import gc
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# globals
# data_path = 'D:\\FIB\\PracticaAPA\\Data\\'
data_path = '../Data/'


def print_df_info(df):
    print('Types:')
    print(df.dtypes)
    print('\nNull values:')
    print(df.isnull().sum())
    print('Memory consumed by dataframe : {} MB\n'.format(df.memory_usage(index=True).sum() / 1024 ** 2))
    print('nrows: {}\n'.format(len(df)))


# TODO
def mca(x):
    return x


def fill_na_gender_knn(members):
    index = members['gender'].isnull()
    x_test = members.ix[index, members.columns != 'gender']
    msno_test = x_test['msno']
    x_test = x_test.ix[:, x_test.columns != 'msno']

    x_train = members.dropna(subset=['gender'])
    y_train = x_train.ix[:, x_train.columns == 'gender']
    x_train = x_train.ix[:, x_train.columns != 'gender']
    msno_train = x_train['msno']
    x_train = x_train.ix[:, x_train.columns != 'msno']

    knn = KNeighborsClassifier(algorithm='ball_tree', n_jobs=-1)
    knn_trained = knn.fit(x_train, y_train.values.ravel())
    new_y = knn_trained.predict(x_test)
    x_test['gender'] = new_y
    x_test['msno'] = msno_test
    x_train['gender'] = y_train
    x_train['msno'] = msno_train
    x_train = x_train.append(x_test)
    return x_train


def fill_na_genre_ids_knn(songs):
    index = songs['genre_ids'].isnull()
    x_test = songs.ix[index, songs.columns != 'genre_ids']
    song_id_test = x_test['song_id']
    x_test = x_test.ix[:, x_test.columns != 'song_id']

    x_train = songs.dropna(subset=['genre_ids'])
    y_train = x_train.ix[:, x_train.columns == 'genre_ids']
    x_train = x_train.ix[:, x_train.columns != 'genre_ids']
    song_id_train = x_train['song_id']
    x_train = x_train.ix[:, x_train.columns != 'song_id']

    knn = KNeighborsClassifier(algorithm='ball_tree', n_jobs=-1)
    knn_trained = knn.fit(x_train, y_train.values.ravel())
    new_y = knn_trained.predict(x_test)
    x_test['genre_ids'] = new_y
    x_test['song_id'] = song_id_test
    x_train['genre_ids'] = y_train
    x_train['song_id'] = song_id_train
    x_train = x_train.append(x_test)
    return x_train


def main():
    # # songs
    df_songs = pd.read_csv(data_path + 'songs.csv')
    df_songs = df_songs.drop(['lyricist'], axis=1)
    df_songs = df_songs.drop(['composer'], axis=1)
    more_freq_language = df_songs['language'].value_counts().idxmax()
    df_songs['language'] = df_songs['language'].fillna(more_freq_language)
    # df_songs = pd.get_dummies(data=df_songs, columns=['language'])
    # df_songs = pd.get_dummies(data=df_songs, columns=['artist_name'], prefix_sep='|')
    print_df_info(df_songs)
    df_songs[['song_id', 'song_length', 'genre_ids', 'language']] = fill_na_genre_ids_knn(
        df_songs[['song_id', 'song_length', 'genre_ids', 'language']])
    df_songs['genre_ids'] = df_songs['genre_ids'].astype('category')
    df_songs['artist_name'] = df_songs['artist_name'].astype('category')
    df_songs['language'] = df_songs['language'].astype('category')
    print_df_info(df_songs)

    # # members
    df_members = (pd.read_csv(data_path + 'members.csv'))[['msno', 'city', 'bd', 'gender']]
    df_members = fill_na_gender_knn(df_members)
    df_members['age_range'] = pd.cut(df_members['bd'], bins=[5, 10, 18, 30, 45, 60, 80])
    more_freq_age_range = df_members['age_range'].value_counts().idxmax()
    df_members['age_range'] = df_members['age_range'].fillna(more_freq_age_range)
    df_members = df_members.drop(['bd'], axis=1)
    df_members['city'] = df_members['city'].astype('category')
    df_members['gender'] = df_members['gender'].astype('category')
    df_members['age_range'] = df_members['age_range'].astype('category')
    print_df_info(df_members)

    # # test
    df_test = pd.read_csv(data_path + 'test.csv')
    df_test = df_test.drop(['source_screen_name'], axis=1)
    print_df_info(df_test)


if __name__ == "__main__":
    main()
