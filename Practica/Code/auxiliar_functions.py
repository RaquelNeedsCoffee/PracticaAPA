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


def data_from_filesMCA():
    print('loading data...')
    warnings.filterwarnings('ignore')
    X_train = pd.read_csv('../Data/preprocessedTrainMCA.csv', compact_ints=True)
    #X_test = pd.read_csv('../Data/preprocessedTestMCA.csv', compact_ints=True)
    X_val = pd.read_csv('../Data/preprocessedValMCA.csv', compact_ints=True)

    y_train = X_train['target']
    #y_test = X_test['target']
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
    return X_train, X_val, y_train, y_val# X_test,y_test,


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


def count_genres_freq(df):
    freq_map = {}
    for g in df['genre_ids']:
        if g is not np.nan:
            song_genres = g.split('|')
            for sg in song_genres:
                if sg in freq_map:
                    freq_map[sg] += 1
                else:
                    freq_map[sg] = 1
    return freq_map


def get_max_genre(song_genres, genres_count_dict):
    song_genres = song_genres.split('|')
    song_genres_dict = {}
    for k in song_genres:
        song_genres_dict[k] = genres_count_dict[k]
    return max(song_genres_dict, key=song_genres_dict.get)


def process_genres(song_genres, genres, genres_count):
    if song_genres is np.nan:
        return np.nan
    sg_list = [g for g in song_genres.split('|') if g in genres]
    sg_dict = {}
    for g in sg_list:
        if g in genres_count:
            sg_dict[g] = genres_count[g]
    if len(sg_dict) <= 0:
        return np.nan
    return max(sg_dict, key=sg_dict.get)


def get_sons_with_only_one_genre(songs):
    print('Remove less common genres that doesn\'t appear in test and limit categories per song to 1:')
    genres_count = count_genres_freq(songs)
    sorted_genres_count = sorted(genres_count.items(), key=operator.itemgetter(1), reverse=True)
    portion_keep_genres = 0.5
    num_genres = math.floor(portion_keep_genres * len(sorted_genres_count))
    genres_train = sorted_genres_count[:num_genres]
    genres_train = list(map(lambda x: x[0], genres_train))
    final_genres = set()
    final_genres.update(genres_train)
    songs['genre_ids'] = songs['genre_ids'].apply(lambda i: process_genres(i, final_genres, genres_count))
    songs['genre_ids'] = songs['genre_ids'].astype('category')
    return songs