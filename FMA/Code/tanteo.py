"""
El objetivo de este archivo es tantear el dataet partiendo del trabajo previo.
"""


import os
import IPython.display as ipd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as skl
import sklearn.utils, sklearn.preprocessing, sklearn.decomposition, sklearn.svm
import librosa
import librosa.display
import ast


import sys

from fma import utils

plt.rcParams['figure.figsize'] = (17, 5)

# Directory where mp3 and the metadata are stored.
AUDIO_DIR = '../Data/fma_small'
METADATA_PATH = '../Data/fma_metadata/'


def load(filepath):

    filename = os.path.basename(filepath)

    if 'features' in filename:
        return pd.read_csv(filepath, index_col=0, header=[0, 1, 2])

    if 'echonest' in filename:
        return pd.read_csv(filepath, index_col=0, header=[0, 1, 2])

    if 'genres' in filename:
        return pd.read_csv(filepath, index_col=0)

    if 'tracks' in filename:
        tracks = pd.read_csv(filepath, index_col=0, header=[0, 1])

        COLUMNS = [('track', 'tags'), ('album', 'tags'), ('artist', 'tags'),
                   ('track', 'genres'), ('track', 'genres_all'),
                   ('track', 'genres_top')]
        for column in COLUMNS:
            tracks[column] = tracks[column].map(ast.literal_eval)

        COLUMNS = [('track', 'date_created'), ('track', 'date_recorded'),
                   ('album', 'date_created'), ('album', 'date_released'),
                   ('artist', 'date_created'), ('artist', 'active_year_begin'),
                   ('artist', 'active_year_end')]
        for column in COLUMNS:
            tracks[column] = pd.to_datetime(tracks[column])

        SUBSETS = ('small', 'medium', 'large')
        tracks['set', 'subset'] = tracks['set', 'subset'].astype(
                'category', categories=SUBSETS, ordered=True)

        COLUMNS = [('track', 'license'), ('artist', 'bio'),
                   ('album', 'type'), ('album', 'information')]
        for column in COLUMNS:
            tracks[column] = tracks[column].astype('category')

        return tracks



# Load metadata and features.
tracks = load(METADATA_PATH + 'tracks.csv')
#tracks = pd.read_csv(METADATA_PATH + 'tracks.csv', index_col=0, header=[0, 1])
genres = utils.load(METADATA_PATH + 'genres.csv')
features = utils.load(METADATA_PATH + 'features.csv')
echonest = utils.load(METADATA_PATH + 'echonest.csv')

# np.testing.assert_array_equal(features.index, tracks.index)
# assert echonest.index.isin(tracks.index).all()

# print('Tracks shape: ', tracks.shape, 'Generes shape: ', genres.shape,
#       'Features shape: ', features.shape, 'Echonest shape: ',  echonest.shape)
print(echonest.head())