import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from scipy.interpolate import interpn
import seaborn as sn
import pandas as pd


# Pandas settings
pd.options.display.max_columns = 10
pd.options.display.max_colwidth = 30


def read_data():
    query_df = pd.read_csv('./data/query_product.csv', encoding='latin1')
    product_df = pd.read_csv('./data/product_descriptions.csv', encoding='latin1')
    train_df = pd.read_csv('./data/train.csv', encoding='latin1')
    return query_df, product_df, train_df


def feature_heatmap(df: pd.DataFrame, x: str, y: str, bin_size):
    x = df[x]
    y = df[y]
    plt.hist2d(x, y, (bin_size, bin_size), cmap='magma')
    plt.colorbar()
    plt.title('Feature heatmap')
    plt.xlabel(x.name)
    plt.ylabel(y.name)
    plt.show()


def box_plot(df, col: str):
    df.boxplot(column=col, by='relevance')
    plt.show()


def pair(df: pd.DataFrame):
    df = df.drop(columns=['id', 'product_uid', 'title', 'query', 'description', 'title_org', 'query_org',
                          'description_org', 'title_non_stem', 'query_non_stem', 'relevance_class'])
    sn.pairplot(df)
    plt.show()


def hist(df: pd.DataFrame, col: str, discrete: bool = False):
    compare_col = 'relevance_class'
    hue_order = ['high', 'low']
    sn.histplot(df[[col, compare_col]], x=col, hue=compare_col, hue_order=hue_order, stat='count', multiple='stack',
                discrete=discrete)
    plt.show()


def explore_dataframe(df: pd.DataFrame):
    # Missing values
    print(df.info())
    print('\n')

    # Integer mean statistics
    print(df.describe())
    print('\n')


def explore():
    query_df, product_df, train_df = read_data()

    # Plot
    # Hists
    hist(train_df, 'word_count_title', discrete=True)
    hist(train_df, 'word_count_description', discrete=True)

    hist(train_df, 'word_count_title', discrete=True)
    hist(train_df, 'word_count_description', discrete=True)

    hist(train_df, 'words_in_common_query_title', discrete=True)
    hist(train_df, 'words_in_common_query_description', discrete=True)

    hist(train_df, 'ratio_in_common_query_title', discrete=False)
    hist(train_df, 'ratio_in_common_query_description', discrete=False)

    # Heatmap
    feature_heatmap(train_df, 'ratio_in_common_query_title', 'relevance', bin_size=10)
    feature_heatmap(train_df, 'ratio_in_common_query_description', 'relevance', bin_size=10)

    feature_heatmap(train_df, 'word_count_title', 'relevance', bin_size=10)

    # box_plot(train_df, 'ratio_in_common_query_title')
    # pair(train_df)


if __name__ == '__main__':
    explore()
