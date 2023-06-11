from typing import Tuple, Any

import pandas as pd
import matplotlib.pyplot as plt


def read_data() -> tuple[pd.DataFrame | Any, pd.DataFrame | Any, pd.DataFrame]:
    query_df = pd.read_csv('./data/query_product.csv', encoding='latin1')
    product_df = pd.read_csv('./data/product_descriptions.csv', encoding='latin1')
    df = pd.merge(query_df, product_df, on='product_uid', how='left')
    return query_df, product_df, df


def unique_values(df: pd.DataFrame):
    for col in df.columns:
        unique = df[col].unique().sum()
        print(f'Column [{col}]: {unique}')


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
    if 'relevance' in df.columns:
        df['relevance'].plot.hist(bins=6, title='relevance')
        plt.show()


def explore():
    q_df, p_df, df = read_data()

    # Query df
    print('<Query df>')
    explore_dataframe(q_df)


if __name__ == '__main__':
    explore()