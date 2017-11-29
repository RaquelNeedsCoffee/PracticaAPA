import gc
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer

# globals
# data_path = 'D:\\FIB\\PracticaAPA\\Data\\'


data_path = '../Data/'


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


def fill_na_gender_knn(members):
	pass


def process_members():
	print('Loading members')
	members = pd.read_csv(data_path + 'members.csv')
	print('members loaded -> loaded {} rows'.format(len(members)))
	print(':----- members -----:')
	print_df_info(members)
	print('Convert columns')
	members['city'] = members['city'].astype(np.uint8)
	members['bd'] = members['bd'].astype(np.uint8)
	# members['gender'] = members['gender'].astype('category')  # to uint in a moment
	members['registered_via'] = members['registered_via'].astype(np.uint8)
	# members['msno'] = members['msno'].astype('category')# no memory reduction
	print('Imput missing "gender" values')
	# TODO: Cambiar more_freq_gender to knn
	more_freq_gender = members['gender'].value_counts().idxmax()
	members['gender'] = members['gender'].fillna(more_freq_gender)
	print('"gender" to numeric')
	members['gender'] = LabelBinarizer().fit_transform(members['gender'])
	members['gender'] = members['gender'].astype(np.uint8)
	print_df_info(members)
	print('\n')
	return members


def process_songs():
	print('Loading songs')
	songs = pd.read_csv(data_path + 'songs.csv')
	print('songs loaded -> loaded {} rows'.format(len(songs)))
	print(':----- songs -----:')
	print_df_info(songs)
	print('Convert columns')
	songs['genre_ids'] = songs['genre_ids'].astype('category')
	songs['lyricist'] = songs['lyricist'].astype('category')
	# songs['language'] = songs['language'].astype('category')  # error later on int conversion
	# songs['song_id'] = songs['song_id'].astype('category')  # no memory reduction
	# songs['artist_name'] = songs['artist_name'].astype('category')  # no memory reduction
	songs['composer'] = songs['composer'].astype('category')  # no memory reduction
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


def process_song_extra():
	print('Loading song_extra')
	song_extra = pd.read_csv(data_path + 'song_extra_info.csv')
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
	song_extra['song_year'] = song_extra['song_year'].fillna(more_freq_song_year)
	print_df_info(song_extra)
	print('\n')
	return song_extra


def final_preprocessing(df):
	pass


print('Loading data...\n')

# # Merge and preprocess train and members into training
df_train = process_train()
df_members = process_members()

df_training = df_train.merge(df_members, on='msno', how='left')
print('merged df_train df_members -> df_training {} rows'.format(len(df_training)))
# df_training['msno'] = df_training['msno'].astype('category')# drop msno later
print_df_info(df_training)

# Drop msno
print('drop msno')
df_training = df_training.drop('msno', axis=1)
print_df_info(df_training)

del df_train
del df_members
gc.collect()

# # Merge and preprocess songs and training
df_songs = process_songs()

df_training = df_training.merge(df_songs, on='song_id', how='left')
print('merged df_training df_songs -> df_training {} rows'.format(len(df_training)))
print_df_info(df_training)
print('Convert columns')
# df_training['song_id'] = df_songs['song_id'].astype('category') # no memory reduction
df_training['language'] = df_training['language'].fillna(0)
df_training['language'] = df_training['language'].astype(np.int8)
df_training['language'] = df_training['language'].replace(0, np.nan)
print_df_info(df_training)

print('Process genres and imput missing values:')
genres_count = count_genres_freq(df_training)
# genres_count = sorted(genres_count.items(), key=operator.itemgetter(1), reverse=True)
max_count_genre = np.nan
if len(genres_count) > 0:
	max_count_genre = max(genres_count, key=genres_count.get)
if max_count_genre is not np.nan:
	df_training['genre_ids'] = df_training['genre_ids'].fillna(max_count_genre)
print('Substitute multiple genres on song by single genre:')
# df_training['genre_ids'].replace #? get_max_genre <- fer replace dels q tinguin multiple genre per un de sol
print_df_info(df_training)

del df_songs
gc.collect()

# # Merge and preprocess song_extra and training
df_song_extra = process_song_extra()

df_training = df_training.merge(df_song_extra, on='song_id', how='left')
print('merged df_training df_song_extra -> df_training {} rows'.format(len(df_training)))
print_df_info(df_training)

print('Drop rows with NaN values of df_training')
df_training = df_training.dropna()
print('merged df_training df_song_extra -> df_training {} rows'.format(len(df_training)))
print_df_info(df_training)

del df_song_extra
del df_training
gc.collect()

print('Done loading...')
