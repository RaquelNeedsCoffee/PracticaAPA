import gc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# global
data_path = '../Data/'
plt.interactive(True)


def extract_info(df):
    nrows = len(df)
    nullsum = df.isnull().sum().rename('null_count')
    nullper100 = nullsum.apply(lambda x: (100*x)/nrows)
    nullper100 = nullper100.rename('null_percentage')
    df_types = df.dtypes.rename('dtypes')
    info_series = pd.concat([nullsum, nullper100, df_types], axis=1)
    return info_series


def print_df_info(df):
    info_series = extract_info(df)
    nrows = len(df)
    print(info_series, '\n')
    print('nrows: {}\n'.format(nrows))
    print('Memory consumed by dataframe : {} MB\n'.format(df.memory_usage(index=True).sum() / 1024 ** 2))


def plot_lost_values_percent(percent_series):
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
    # plot figure size
    plt.figure(figsize=(30, 23))
    # remove top and right and ensure it shows only left and bottom framelines.
    ax = plt.subplot(111)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    # limit y range
    plt.ylim(0, 50)
    # ticks font and change yticks names
    ticks_range_values_y = range(0, 50, 10)
    plt.yticks(ticks_range_values_y, [str(x) + "%" for x in ticks_range_values_y], fontsize=14)
    plt.xticks(fontsize=14)
    # yticks lines across plot to help readability
    max_range_x = len(percent_series.values) + 1
    range_values_y = range(5, 50, 5)
    for y in range_values_y:
        plt.plot(range(0, max_range_x), [y] * len(range(0, max_range_x)), "--", lw=0.5, color="black", alpha=0.3)

    plt.title("Dataset missing values per feature percentage")
    percent_series.plot.bar(color=tableau20)


def main():
    # load data
    print("Loading data:")
    print("\t-- loading train.csv --")
    df_train = pd.read_csv(data_path + 'train.csv', nrows=None, dtype={'target': np.uint8})
    print_df_info(df_train)
    print("\t-- loading members.csv --")
    df_members = pd.read_csv(data_path + 'members.csv')
    print_df_info(df_members)
    print("\t-- loading songs.csv --")
    df_songs = pd.read_csv(data_path + 'songs.csv', nrows=None)
    print_df_info(df_songs)
    print("\t-- loading song_extra_info.csv --")
    df_song_extra = pd.read_csv(data_path + 'song_extra_info.csv', nrows=None)
    print_df_info(df_song_extra)
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
    plot_lost_values_percent(extract_info(df_merged)['null_percentage'])


if __name__ == "__main__":
    main()
