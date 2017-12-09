import gc
import operator
import math
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.neighbors import KNeighborsClassifier

# globals
data_path = 'D:\\FIB\\PracticaAPA\\Data\\'

# data_path = '../Data/'


def split_isrc(isrc):
    country_code, registrant_code, song_year, isrc_id = np.nan, np.nan, np.nan, np.nan
    if isinstance(isrc, str) and len(isrc) >= 12:
        country_code = isrc[0:2]
        registrant_code = isrc[2:5]
        yy = int(isrc[5:7])
        if yy > 20:
            song_year = 1900 + yy
        else:
            song_year = 2000 + yy
            isrc_id = isrc[7:12]
    return country_code, registrant_code, song_year, isrc_id


'''
A implementar:
    IT 1:
        -> Contar generos individuales y anadirlos al dict
     -> Generar lista canciones con 2 o mas generos
    IT 2:
        -> Canciones con 2 o mas generos:
        -> Coger 2 generos (X, Y) mas freq entre los generos de la cancion, anadir entrada X|Y al dict
        -> Generar lista canciones con 3 o mas generos
    IT 3:
        -> Canciones con 3 o mas generos:
        -> Mirar combinacion X|Y mas freq, mirar genero Z mas freq del resto de generos de la cancion (Z != X ^ Z != Y)
        -> Generar lista canciones con 4 o mas generos
    ETC (hasta que la lista generada sea nula)
WHY?
Puede que sea mas frequente una combinacion de generos que un genero individual 
(ejemplo: canciones pop rock -> dos generos, pero seguramente mas habitual que musica folclorica noruega)
Asi podriamos discriminar manteniendo lo mas frequente (ej: una cancion que tenga "pop rock funk rap" 
-> lo mas frequente de las posibles combinaciones es pop rock -> la dejamos como pop rock (y nos cargamos el resto)
Por otro lado podria ser util para interpolar valores si es necesario (puede, aun no se que resultado dara esto, 
falta implementarlo y comprobar si sirve de algo).
'''


# primera iteracion hecha (falta implementar lo anotado arriva)


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


def get_test_genres(df):
    genres_set = set()
    for g in df['genre_ids']:
        if g is not np.nan:
            genres_set.update(g.split('|'))
    return genres_set


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


def print_df_info(df):
    print('Types:')
    print(df.dtypes)
    print('\nNull values:')
    print(df.isnull().sum())
    print('Memory consumed by dataframe : {} MB\n'.format(df.memory_usage(index=True).sum() / 1024 ** 2))


def process_train():
    print('Loading train')
    train = pd.read_csv(data_path + 'train.csv', nrows=None, dtype={'target': np.uint8})
    print('train loaded -> loaded {} rows'.format(len(train)))
    print(':----- train -----:')
    print_df_info(train)
    print('Convert columns')
    # train['msno'] = train['msno'].astype('category')  # drop after
    train['source_system_tab'] = train['source_system_tab'].astype('category')
    train['source_screen_name'] = train['source_screen_name'].astype('category')
    train['source_type'] = train['source_type'].astype('category')
    # train['song_id'] = train['song_id'].astype('category')  # crash merge df_songs
    print('Drop "NaN" rows of "source_system_tab" and "source_type"')  # low ammount of nan values (20K of 7M)
    train = train.dropna(subset=['source_system_tab'])
    train = train.dropna(subset=['source_type'])
    print('Imput missing "source_screen_name" values')
    more_freq_source_screen_name = train['source_screen_name'].value_counts().idxmax()
    train['source_screen_name'] = train['source_screen_name'].fillna(more_freq_source_screen_name)
    print_df_info(train)
    print('\n')
    return train


def process_test():
    print('Loading test')
    test = pd.read_csv(data_path + 'test.csv', nrows=None, dtype={'target': np.uint8})
    print('test loaded -> loaded {} rows'.format(len(test)))
    print(':----- test -----:')
    print_df_info(test)
    print('Convert columns')
    test['source_system_tab'] = test['source_system_tab'].astype('category')
    test['source_screen_name'] = test['source_screen_name'].astype('category')
    test['source_type'] = test['source_type'].astype('category')
    print('Imput missing "source_system_tab", "source_screen_name" and "source_type" values')
    more_freq_source_system_tab = test['source_system_tab'].value_counts().idxmax()
    test['source_system_tab'] = test['source_system_tab'].fillna(more_freq_source_system_tab)
    more_freq_source_screen_name = test['source_screen_name'].value_counts().idxmax()
    test['source_screen_name'] = test['source_screen_name'].fillna(more_freq_source_screen_name)
    more_freq_source_type = test['source_type'].value_counts().idxmax()
    test['source_type'] = test['source_type'].fillna(more_freq_source_type)
    print_df_info(test)
    print('test -> {} rows'.format(len(test)))
    print('\n')
    return test


def fill_na_gender_knn(members):
    index = members['gender'].isnull()
    X_test = members.ix[index, members.columns != 'gender']
    msno_test = X_test['msno']
    X_test = X_test.ix[:, X_test.columns != 'msno']

    X_train = members.dropna(subset=['gender'])
    y_train = X_train.ix[:, X_train.columns == 'gender']
    X_train = X_train.ix[:, X_train.columns != 'gender']
    msno_train = X_train['msno']
    X_train = X_train.ix[:, X_train.columns != 'msno']

    knn = KNeighborsClassifier(algorithm='ball_tree', n_jobs=-1)
    knn_trained = knn.fit(X_train, y_train.values.ravel())
    new_y = knn_trained.predict(X_test)
    X_test['gender'] = new_y
    X_test['msno'] = msno_test
    X_train['gender'] = y_train
    X_train['msno'] = msno_train
    X_train = X_train.append(X_test)
    return X_train


def process_members():
    print('Loading members')
    members = pd.read_csv(data_path + 'members.csv')
    # remove_less_common(members, threshold=100)
    print('members loaded -> loaded {} rows'.format(len(members)))
    print(':----- members -----:')
    print_df_info(members)
    print('Convert columns')
    members['city'] = members['city'].astype(np.uint8)
    # cambio bd a age_range
    members['bd'] = members['bd'].astype(np.uint8)
    # members['gender'] = members['gender'].astype('category')  # to uint in a moment
    members['registered_via'] = members['registered_via'].astype(np.uint8)
    # members['msno'] = members['msno'].astype('category')# no memory reduction
    print('Imput missing "gender" values')
    members = fill_na_gender_knn(members)
    print('"gender" to numeric')
    members['gender'] = LabelBinarizer().fit_transform(members['gender'])
    members['gender'] = members['gender'].astype(np.uint8)
    members['age_range'] = pd.cut(members['bd'], bins=[5, 10, 18, 30, 45, 60, 80])
    more_freq_age_range = members['age_range'].value_counts().idxmax()
    members['age_range'] = members['age_range'].fillna(more_freq_age_range)
    members.drop(['bd'], axis=1)
    print_df_info(members)
    print('\n')
    return members


def process_songs():
    print('Loading songs')
    songs = pd.read_csv(data_path + 'songs.csv')
    remove_less_common(songs)
    print('songs loaded -> loaded {} rows'.format(len(songs)))
    print(':----- songs -----:')
    print_df_info(songs)
    print('Convert columns')
    # songs['genre_ids'] = songs['genre_ids'].astype('category')
    songs['lyricist'] = songs['lyricist'].astype('category')
    # songs['language'] = songs['language'].astype('category')  # error later on int conversion
    # songs['song_id'] = songs['song_id'].astype('category')  # no memory reduction
    # songs['artist_name'] = songs['artist_name'].astype('category')  # no memory reduction
    songs['composer'] = songs['composer'].astype('category')  # no memory reduction
    # drop col 'name'
    # songs = songs.drop('name', axis=1)
    print('Drop "NaN" rows of "language"')  # low ammount of nan values (1 of 2M)
    songs = songs.dropna(subset=['language'])
    print('Add category "no_lyricist" into "lyricist" categories')  # a lot of NaN values
    songs['lyricist'] = songs['lyricist'].cat.add_categories(['no_lyricist'])
    songs['lyricist'] = songs['lyricist'].fillna('no_lyricist')
    print('Add category "no_composer" into "composer" categories')  # a lot of NaN values
    songs['composer'] = songs['composer'].cat.add_categories(['no_composer'])
    songs['composer'] = songs['composer'].fillna('no_composer')
    print_df_info(songs)
    print('\n')
    return songs


def remove_less_common(df, threshold=200):
    pass
    # # Anything that occurs less than this will be removed.
    # value_counts = df.stack().value_counts()  # Entire DataFrame
    # to_remove = value_counts[value_counts <= threshold].index
    # df.replace(to_remove, np.nan, inplace=True)


def process_song_extra():
    print('Loading song_extra')
    song_extra = pd.read_csv(data_path + 'song_extra_info.csv')
    remove_less_common(song_extra)
    print('song_extra loaded -> loaded {} rows'.format(len(song_extra)))
    print(':----- song_extra -----:')
    print_df_info(song_extra)
    # print('Convert columns')
    # song_extra['song_id'] = song_extra['song_id'].astype('category')# no memory reduction
    # song_extra['name'] = song_extra['name'].astype('category')# no memory
    # print_df_info(song_extra)
    print('Spliting isrc')
    # don't want isrc_id
    song_extra['country_code'], song_extra['registrant_code'], song_extra['song_year'], _ = \
        zip(*song_extra['isrc'].apply(lambda i: split_isrc(i)))
    # drop col 'isrc'
    song_extra = song_extra.drop('isrc', axis=1)
    print('Drop "NaN" rows of "name"')  # low ammount of nan values (2 of 2M)
    song_extra = song_extra.dropna(subset=['name'])
    print('Imput missing "country_code", "registrant_code" and "song_year" values')
    more_freq_country_code = song_extra['country_code'].value_counts().idxmax()
    song_extra['country_code'] = song_extra['country_code'].fillna(more_freq_country_code)
    more_freq_registrant_code = song_extra['registrant_code'].value_counts().idxmax()
    song_extra['registrant_code'] = song_extra['registrant_code'].fillna(more_freq_registrant_code)
    more_freq_song_year = song_extra['song_year'].value_counts().idxmax()
    song_extra['song_year'] = song_extra['song_year'].fillna(more_freq_song_year).astype(np.uint16)
    print_df_info(song_extra)
    print('\n')
    return song_extra


def final_preprocessing(df):
    pass


def main():
    print('Loading data...\n')

    # Merge and preprocess train and members into training
    df_train = process_train()
    # # test
    df_test = process_test()

    df_members = process_members()

    df_training = df_train.merge(df_members, on='msno', how='left')
    print('merged df_train df_members -> df_training {} rows'.format(len(df_training)))
    print_df_info(df_training)

    # # test
    df_test = df_test.merge(df_members, on='msno', how='left')
    print('merged df_test df_members -> df_test {} rows'.format(len(df_test)))
    print_df_info(df_test)

    del df_train
    del df_members
    gc.collect()

    # Drop msno
    print('drop msno')
    df_training = df_training.drop('msno', axis=1)
    print_df_info(df_training)
    # # test
    print('drop msno test')
    df_test = df_test.drop('msno', axis=1)
    print_df_info(df_test)

    # # Merge and preprocess songs and training
    df_songs = process_songs()

    # # test
    df_test = df_test.merge(df_songs, on='song_id', how='left')
    print('merged df_test df_songs -> df_test {} rows'.format(len(df_test)))
    print_df_info(df_test)
    print('Convert columns test')
    df_test['language'] = df_test['language'].astype('category')
    print_df_info(df_test)

    df_training = df_training.merge(df_songs, on='song_id', how='left')
    print('merged df_training df_songs -> df_training {} rows'.format(len(df_training)))
    print_df_info(df_training)
    print('Convert columns')
    df_training['language'] = df_training['language'].astype('category')
    print_df_info(df_training)

    del df_songs
    gc.collect()

    print('Remove less common genres that doesn\'t appear in test and limit categories per song to 1:')
    genres_count = count_genres_freq(df_training)
    sorted_genres_count = sorted(genres_count.items(), key=operator.itemgetter(1), reverse=True)
    portion_keep_genres = 0.5
    num_genres = math.floor(portion_keep_genres * len(sorted_genres_count))
    genres_train = sorted_genres_count[:num_genres]
    genres_train = list(map(lambda x: x[0], genres_train))
    final_genres = set()
    final_genres.update(get_test_genres(df_test))  # test
    final_genres.update(genres_train)
    df_training['genre_ids'] = df_training['genre_ids'].apply(lambda i: process_genres(i, final_genres, genres_count))
    df_training['genre_ids'] = df_training['genre_ids'].astype('category')
    print_df_info(df_training)

    print('Imput missing values:')
    max_count_genre = np.nan
    if len(genres_count) > 0:
        max_count_genre = max(genres_count, key=genres_count.get)
    if max_count_genre is not np.nan:
        df_training['genre_ids'] = df_training['genre_ids'].fillna(max_count_genre)
    print_df_info(df_training)

    # # test
    print('Limit categories per song to 1 in test:')
    df_test['genre_ids'] = df_test['genre_ids'].apply(lambda i: process_genres(i, final_genres, genres_count))
    print_df_info(df_test)
    print('Imput missing values test:')
    if max_count_genre is not np.nan:
        df_test['genre_ids'] = df_test['genre_ids'].fillna(max_count_genre)
    print_df_info(df_test)

    # # Merge and preprocess song_extra and training
    df_song_extra = process_song_extra()

    # # test
    df_test = df_test.merge(df_song_extra, on='song_id', how='left')
    print('merged df_test df_song_extra -> df_test {} rows'.format(len(df_test)))
    print_df_info(df_test)
    print('Imput missing values of df_test:')
    song_length_median = df_test['song_length'].mean()
    print(song_length_median)
    df_test['song_length'] = df_test['song_length'].fillna(song_length_median)
    more_freq_artist_name = df_test['artist_name'].value_counts().idxmax()
    df_test['artist_name'] = df_test['artist_name'].fillna(more_freq_artist_name)
    more_freq_composer = df_test['composer'].value_counts().idxmax()
    df_test['composer'] = df_test['composer'].fillna(more_freq_composer)
    more_freq_lyricist = df_test['lyricist'].value_counts().idxmax()
    df_test['lyricist'] = df_test['lyricist'].fillna(more_freq_lyricist)
    more_freq_language = df_test['language'].value_counts().idxmax()
    df_test['language'] = df_test['language'].fillna(more_freq_language)
    more_freq_name = df_test['name'].value_counts().idxmax()
    df_test['name'] = df_test['name'].fillna(more_freq_name)
    more_freq_country_code = df_test['country_code'].value_counts().idxmax()
    df_test['country_code'] = df_test['country_code'].fillna(more_freq_country_code)
    more_freq_registrant_code = df_test['registrant_code'].value_counts().idxmax()
    df_test['registrant_code'] = df_test['registrant_code'].fillna(more_freq_registrant_code)
    more_freq_song_year = df_test['song_year'].value_counts().idxmax()
    df_test['song_year'] = df_test['song_year'].fillna(more_freq_song_year)
    print_df_info(df_test)

    df_training = df_training.merge(df_song_extra, on='song_id', how='left')
    print('merged df_training df_song_extra -> df_training {} rows'.format(len(df_training)))
    print_df_info(df_training)

    del df_song_extra
    gc.collect()

    print('Drop rows with NaN values of df_training')
    df_training = df_training.dropna()
    print('merged df_training df_song_extra -> df_training {} rows'.format(len(df_training)))
    df_training.drop(columns=['song_id', 'bd', 'lyricist'])
    print_df_info(df_training)
    df_training = pd.DataFrame(df_training)
    print('writing csv')
    df_training.to_csv(data_path + 'def_training.csv')
    df_test.to_csv(data_path + 'def_test.csv')
    final_preprocessing(df_training)

    del df_training
    del df_test
    gc.collect()

    print('Done loading...')


if __name__ == "__main__":
    main()
