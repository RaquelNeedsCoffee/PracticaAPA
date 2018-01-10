import gc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# global
data_path = '../Data/'
plt.interactive(True)


def print_df_info(df):
    print('Types:')
    print(df.dtypes)
    print('\nNull values:')
    print(df.isnull().sum())
    print('Memory consumed by dataframe : {} MB\n'.format(df.memory_usage(index=True).sum() / 1024 ** 2))
    print('nrows: {}\n'.format(len(df)))


def lost_values_info(df):
    sm = df.isnull().sum()

    # These are the "Tableau 20" colors as RGB.
    tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
                 (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
                 (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
                 (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
                 (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

    # Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.
    for i in range(len(tableau20)):
        r, g, b = tableau20[i]
        tableau20[i] = (r / 255., g / 255., b / 255.)
    plt.figure(figsize=(20, 15))
    ax = plt.subplot(111)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    max_range_x = len(sm.values) + 1
    # range_values_y = range(10000, 80001, 10000)
    range_values_y = range(500000, 3000001, 500000)
    for y in range_values_y:
        plt.plot(range(0, max_range_x), [y] * len(range(0, max_range_x)), "--", lw=0.5, color="black", alpha=0.3)
    plt.tick_params(axis="both", which="both", bottom="off", top="off",
                    labelbottom="on", left="off", right="off", labelleft="on")
    plt.title("Missing values per feature in the dataset")
    sm.plot.bar(color=tableau20)


def main():
    # load data
    print("Loading data:")
    print("\t-- loading train.csv --")
    df_train = pd.read_csv(data_path + 'train.csv', nrows=100000, dtype={'target': np.uint8})
    print("\t-- loading members.csv --")
    df_members = pd.read_csv(data_path + 'members.csv')
    print("\t-- loading songs.csv --")
    df_songs = pd.read_csv(data_path + 'songs.csv', nrows=100000)
    print("\t-- loading song_extra_info.csv --")
    df_song_extra = pd.read_csv(data_path + 'song_extra_info.csv', nrows=100000)
    print("Data loaded")

    # Grafica lost values
    print("merge")
    df_merged = df_train.merge(df_members, on='msno', how='left')
    del df_train, df_members
    gc.collect()
    df_merged = df_merged.merge(df_songs, on='song_id', how='left')
    del df_songs
    gc.collect()
    df_merged = df_merged.merge(df_song_extra, on='song_id', how='left')
    del df_song_extra
    gc.collect()
    print("fmerge")
    print_df_info(df_merged)
    lost_values_info(df_merged)


if __name__ == "__main__":
    main()
