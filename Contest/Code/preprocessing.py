"""
En este archivo mezclo los datasets iniciales y les cambio el formato a algunas variables para que pete menos.
"""

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import gc


df_train = pd.read_csv('../Data/train.csv')
df_test = pd.read_csv('../Data/test.csv')
df_members = pd.read_csv('../Data/members.csv')
df_songs = pd.read_csv('../Data/songs.csv')
# df_sample = pd.read_csv('../input/sample_submission.csv')

type(df_train.iloc[1])

mem = df_train.memory_usage(index=True).sum()
print("Memory consumed by train dataframe : {} MB".format(mem / 1024 ** 2))

df_train['target'] = df_train['target'].astype(np.int8)
df_test['id'] = df_test['id'].astype(np.int32)

mem = df_train.memory_usage(index=True).sum()
print("Memory consumed by train dataframe : {} MB".format(mem / 1024 ** 2))

mem = df_members.memory_usage(index=True).sum()
print("\nMemory consumed by members dataframe : {} MB".format(mem / 1024 ** 2))

df_members['city'] = df_members['city'].astype(np.int8)
df_members['bd'] = df_members['bd'].astype(np.int16)
df_members['registered_via'] = df_members['registered_via'].astype(np.int8)
df_members['registration_init_time'] = df_members['registration_init_time'].astype(np.int32)
df_members['expiration_date'] = df_members['expiration_date'].astype(np.int32)

mem = df_members.memory_usage(index=True).sum()
print("Memory consumed by members dataframe : {} MB".format(mem / 1024 ** 2))

mem = df_songs.memory_usage(index=True).sum()
print("\nMemory consumed by songs dataframe : {} MB".format(mem / 1024 ** 2))

df_songs['song_length'] = df_songs['song_length'].astype(np.int32)

# Since language column contains Nan values we will convert it to 0,
# After converting the type of the column we will revert it back to nan
df_songs['language'] = df_songs['language'].fillna(0)
df_songs['language'] = df_songs['language'].astype(np.int8)
df_songs['language'] = df_songs['language'].replace(0, np.nan)

mem = df_songs.memory_usage(index=True).sum()
print("Memory consumed by songs dataframe : {} MB".format(mem / 1024 ** 2))

train = df_train.merge(df_members, on='msno', how='left')
test = df_test.merge(df_members, on='msno', how='left')

train = train.merge(df_songs[['song_id', 'song_length', 'artist_name', 'genre_ids', 'language']], on='song_id')
test = test.merge(df_songs[['song_id', 'song_length', 'artist_name', 'genre_ids', 'language']], on='song_id')

del df_train
del df_test
del df_songs
del df_members
gc.collect()

# Removing rows having missing values in msno and target ---
train = train[pd.notnull(train['msno'])]
train = train[pd.notnull(train['target'])]

# Returning values to int to save memory
train['target'] = train['target'].astype(np.int8)
test['id'] = test['id'].astype(np.int32)

train['city'] = train['city'].astype(np.int8)
train['bd'] = train['bd'].astype(np.int16)
train['registered_via'] = train['registered_via'].astype(np.int8)
train['registration_init_time'] = train['registration_init_time'].astype(np.int32)
train['expiration_date'] = train['expiration_date'].astype(np.int32)

test['city'] = test['city'].astype(np.int8)
test['bd'] = test['bd'].astype(np.int16)
test['registered_via'] = test['registered_via'].astype(np.int8)
test['registration_init_time'] = test['registration_init_time'].astype(np.int32)
test['expiration_date'] = test['expiration_date'].astype(np.int32)

train['song_length'] = train['song_length'].astype(np.int32)
# Since language column contains Nan values we will convert it to 0,
# After converting the type of the column we will revert it back to nan
train['language'] = train['language'].fillna(0)
train['language'] = train['language'].astype(np.int8)
train['language'] = train['language'].replace(0, np.nan)

test['song_length'] = test['song_length'].astype(np.int32)
# Since language column contains Nan values we will convert it to 0,
# After converting the type of the column we will revert it back to nan
test['language'] = test['language'].fillna(0)
test['language'] = test['language'].astype(np.int8)
test['language'] = test['language'].replace(0, np.nan)

#############################################################################
################## Data Cleaning ############################################
#############################################################################

date_cols = ['registration_init_time', 'expiration_date']
for col in date_cols:
    train[col] = pd.to_datetime(train[col])
    test[col] = pd.to_datetime(test[col])


def check_missing_values(df):
    print('Are missing values: ', df.isnull().values.any())
    if df.isnull().values.any() == True:
        columns_with_Nan = df.columns[df.isnull().any()].tolist()
        print('Columns with nan', columns_with_Nan)
        for col in columns_with_Nan:
            print("%s : number of nans %d" % (col, df[col].isnull().sum()))


check_missing_values(train)
check_missing_values(test)
