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


print('Loading data...')
print('\n\n')

print('Loading df_train')
df_train = pd.read_csv(data_path + 'train.csv', nrows=1000000, dtype={'target': np.uint8})
print('df_train loaded -> loaded {} rows'.format(len(df_train)))
print(':----- df_train -----:')
print(df_train.dtypes)
print('Null values:')
print(df_train.isnull().sum())
print('Memory consumed by dataframe : {} MB'.format(df_train.memory_usage(index=True).sum() / 1024 ** 2))
print('Convert columns - categories')
df_train['msno'] = df_train['msno'].astype('category')
df_train['song_id'] = df_train['song_id'].astype('category')
df_train['source_system_tab'] = df_train['source_system_tab'].astype('category')
df_train['source_screen_name'] = df_train['source_screen_name'].astype('category')
df_train['source_type'] = df_train['source_type'].astype('category')
print(df_train.dtypes)
print('Memory consumed by dataframe : {} MB'.format(df_train.memory_usage(index=True).sum() / 1024 ** 2))
print('\n\n')
del df_train
gc.collect()

print('Loading df_songs')
df_songs = pd.read_csv(data_path + 'songs.csv')
print('df_songs loaded -> loaded {} rows'.format(len(df_songs)))
print(':----- df_songs -----:')
print(df_songs.dtypes)
print('Null values:')
print(df_songs.isnull().sum())
print('Memory consumed by dataframe : {} MB'.format(df_songs.memory_usage(index=True).sum() / 1024 ** 2))
print('Convert columns - categories')
df_songs['song_id'] = df_songs['song_id'].astype('category')
df_songs['genre_ids'] = df_songs['genre_ids'].astype('category')
df_songs['artist_name'] = df_songs['artist_name'].astype('category')
df_songs['composer'] = df_songs['composer'].astype('category')
df_songs['lyricist'] = df_songs['lyricist'].astype('category')
df_songs['language'] = df_songs['language'].astype('category')
print(df_songs.dtypes)
print('Memory consumed by dataframe : {} MB'.format(df_songs.memory_usage(index=True).sum() / 1024 ** 2))
print('\n\n')
del df_songs
gc.collect()

print('Loading df_song_extra')
df_song_extra = pd.read_csv(data_path + 'song_extra_info.csv')
print('df_song_extra loaded -> loaded {} rows'.format(len(df_song_extra)))
# df_song_extra['song_year'] = (df_song_extra['isrc'].apply(lambda i: split_isrc(i)[2]))
print(':----- df_song_extra -----:')
print(df_song_extra.dtypes)
print('Null values:')
print(df_song_extra.isnull().sum())
print('Memory consumed by dataframe : {} MB'.format(df_song_extra.memory_usage(index=True).sum() / 1024 ** 2))
print('Convert columns - categories')
# df_song_extra['song_id'] = df_song_extra['song_id'].astype('category')
df_song_extra['name'] = df_song_extra['name'].astype('category')
print(df_song_extra.dtypes)
print('Memory consumed by dataframe : {} MB'.format(df_song_extra.memory_usage(index=True).sum() / 1024 ** 2))
print('\n\n')
del df_song_extra
gc.collect()

print('Loading df_members')
df_members = pd.read_csv(data_path + 'members.csv')
print('df_members loaded -> loaded {} rows'.format(len(df_members)))
print(':----- df_members -----:')
print(df_members.dtypes)
print('Null values:')
print(df_members.isnull().sum())
print('Memory consumed by dataframe : {} MB'.format(df_members.memory_usage(index=True).sum() / 1024 ** 2))
print('Convert columns - categories')
df_members['msno'] = df_members['msno'].astype('category')
df_members['city'] = df_members['city'].astype('category')
df_members['bd'] = df_members['bd'].astype(np.uint8)
df_members['gender'] = df_members['gender'].astype('category')
df_members['registered_via'] = df_members['registered_via'].astype('category')
print(df_members.dtypes)
print('Memory consumed by dataframe : {} MB'.format(df_members.memory_usage(index=True).sum() / 1024 ** 2))
print('\n\n')
del df_members
gc.collect()

print('Done loading...')
