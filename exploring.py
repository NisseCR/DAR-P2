import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from scipy.interpolate import interpn
import pandas as pd


# Pandas settings
pd.options.display.max_columns = 10
pd.options.display.max_colwidth = 30


def read_data():
    query_df = pd.read_csv('./data/query_product.csv', encoding='latin1')
    product_df = pd.read_csv('./data/product_descriptions.csv', encoding='latin1')
    clean_df = pd.read_csv('./data/data.csv', encoding='latin1')
    return query_df, product_df, clean_df


def unique_values(df: pd.DataFrame):
    for col in df.columns:
        unique = df[col].unique().sum()
        print(f'Column [{col}]: {unique}')


def text_distribution(ser: pd.Series):
    data = ser.value_counts().reset_index()
    print(data)
    data.plot.bar(use_index=True, y='count')
    plt.show()


def feature_heatmap(x: pd.Series, y: pd.Series, bin_size):
    plt.hist2d(x, y, (bin_size, bin_size), cmap=plt.cm.jet)
    plt.colorbar()
    plt.title('Feature heatmap')
    plt.xlabel(x.name)
    plt.ylabel(y.name)


def box_plot(df, col: str):
    df.boxplot(column='relevance', by=col)


def explore_dataframe(df: pd.DataFrame):
    # Missing values
    print(df.info())
    print('\n')

    # Integer mean statistics
    print(df.describe())
    print('\n')

    # Object unique statistics
    print(df.describe(include=['object', 'bool']))

    # Distribution
    text_distribution(df['search_term'])


def explore():
    query_df, product_df, clean_df = read_data()

    # Preprocessed data
    feature_heatmap(clean_df['ratio_in_common'], clean_df['relevance'], bin_size=5)
    box_plot(clean_df, 'complete_ratio')

    # Render
    plt.show()


if __name__ == '__main__':
    explore()
