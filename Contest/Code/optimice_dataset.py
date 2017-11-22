import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import gc


def optimice(train):
    mem = train.memory_usage(index=True).sum()
    print("\nMemory consumed by dataframe : {} MB".format(mem / 1024 ** 2))

    # Removing rows having missing values in msno and target ---
    train = train[pd.notnull(train['msno'])]
    #train = train[pd.notnull(train['target'])]

    # Returning values to int to save memory
    #train['target'] = train['target'].astype(np.int8)

    train['city'] = train['city'].astype(np.int8)
    train['bd'] = train['bd'].astype(np.int16)
    train['registered_via'] = train['registered_via'].astype(np.int8)
    train['registration_init_time'] = train['registration_init_time'].astype(np.int32)
    train['expiration_date'] = train['expiration_date'].astype(np.int32)

    train['song_length'] = train['song_length'].astype(np.int32)
    # Since language column contains Nan values we will convert it to 0,
    # After converting the type of the column we will revert it back to nan
    train['language'] = train['language'].fillna(0)
    train['language'] = train['language'].astype(np.int8)
    train['language'] = train['language'].replace(0, np.nan)

    mem = train.memory_usage(index=True).sum()
    print("\nMemory consumed by dataframe : {} MB".format(mem / 1024 ** 2))

    return train
