import gc
import pandas as pd
import numpy as np

# globals
data_path = '../Data/'


def split_isrc(isrc):
    cc, xxx, yy, nnnnn = np.nan, np.nan, np.nan, np.nan
    if isinstance(isrc, str):
        cc = isrc[0:2]
        xxx = isrc[2:5]
        if int(isrc[5:7]) > 20:
            yy = 1900 + int(isrc[5:7])
        else:
            yy = 2000 + int(isrc[5:7])
        nnnnn = isrc[7:12]
    return cc, xxx, yy, nnnnn


def print_df_info(df):
    # print('Types:')
    # print(df.dtypes)
    # print('Null values:')
    # print(df.isnull().sum())
    print('Memory consumed by dataframe : {} MB\n\n'.format(df.memory_usage(index=True).sum() / 1024 ** 2))


print('Loading data...')
print('\n\n')

print('Loading df_train')
df_train = pd.read_csv(data_path + 'train.csv', nrows=None, dtype={'target': np.uint8})
print('df_train loaded -> loaded {} rows'.format(len(df_train)))
print(':----- df_train -----:')
print_df_info(df_train)
print('Convert columns')
df_train['msno'] = df_train['msno'].astype('category')
df_train['source_system_tab'] = df_train['source_system_tab'].astype('category')
df_train['source_screen_name'] = df_train['source_screen_name'].astype('category')
df_train['source_type'] = df_train['source_type'].astype('category')
# df_train['song_id'] = df_train['song_id'].astype('category')# crash merge df_songs
print_df_info(df_train)
print('\n\n')

print('Loading df_members')
df_members = pd.read_csv(data_path + 'members.csv')
print('df_members loaded -> loaded {} rows'.format(len(df_members)))
print(':----- df_members -----:')
print_df_info(df_members)
print('Convert columns')
df_members['city'] = df_members['city'].astype(np.uint8)
df_members['bd'] = df_members['bd'].astype(np.uint8)
df_members['gender'] = df_members['gender'].astype('category')
df_members['registered_via'] = df_members['registered_via'].astype(np.uint8)
# df_members['msno'] = df_members['msno'].astype('category')# no memory reduction
print_df_info(df_members)
print('\n\n')

df_training = df_train.merge(df_members, on='msno', how='left')
print('merged df_train df_members -> df_training {} rows'.format(len(df_training)))
df_training['msno'] = df_training['msno'].astype('category')
print_df_info(df_training)

del df_train
del df_members
gc.collect()

print('Loading df_songs')
df_songs = pd.read_csv(data_path + 'songs.csv')
print('df_songs loaded -> loaded {} rows'.format(len(df_songs)))
print(':----- df_songs -----:')
print_df_info(df_songs)
print('Convert columns')
df_songs['genre_ids'] = df_songs['genre_ids'].astype('category')
df_songs['lyricist'] = df_songs['lyricist'].astype('category')
# df_songs['language'] = df_songs['language'].astype('category')
# df_songs['song_id'] = df_songs['song_id'].astype('category')# no memory reduction
# df_songs['artist_name'] = df_songs['artist_name'].astype('category')# no memory reduction
# df_songs['composer'] = df_songs['composer'].astype('category')# no memory reduction
print_df_info(df_songs)
print('\n\n')


df_training = df_training.merge(df_songs, on='song_id', how='left')
print('merged df_training df_songs -> df_training {} rows'.format(len(df_training)))
print_df_info(df_training)
print('Convert columns')
# df_training['song_id'] = df_songs['song_id'].astype('category')# no memory reduction
df_training['language'] = df_training['language'].fillna(0)
df_training['language'] = df_training['language'].astype(np.int8)
df_training['language'] = df_training['language'].replace(0, np.nan)
print_df_info(df_training)
print('\n\n')

del df_songs
gc.collect()

print('Loading df_song_extra')
df_song_extra = pd.read_csv(data_path + 'song_extra_info.csv')
print('df_song_extra loaded -> loaded {} rows'.format(len(df_song_extra)))
# df_song_extra['song_year'] = (df_song_extra['isrc'].apply(lambda i: split_isrc(i)[2]))
print(':----- df_song_extra -----:')
print_df_info(df_song_extra)
print('Convert columns')
# df_song_extra['song_id'] = df_song_extra['song_id'].astype('category')# no memory reduction
# df_song_extra['name'] = df_song_extra['name'].astype('category')# no memory reduction
print_df_info(df_song_extra)
print('\n\n')

df_training = df_training.merge(df_song_extra, on='song_id', how='left')
print('merged df_training df_song_extra -> df_training {} rows'.format(len(df_training)))
print_df_info(df_training)

del df_song_extra
del df_training
gc.collect()

print('\n\n')
print('Done loading...')
