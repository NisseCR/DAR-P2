import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib_venn import venn2, venn2_circles
from scipy.interpolate import interpn
import seaborn as sn
import pandas as pd

# Pandas settings
pd.options.display.max_columns = 10
pd.options.display.max_colwidth = 50


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
    df.boxplot(column='relevance', by=col)
    plt.show()


def pair(df: pd.DataFrame):
    df = df.drop(columns=['id', 'product_uid', 'title', 'query', 'description', 'title_org', 'query_org',
                          'description_org', 'title_non_stem', 'query_non_stem', 'relevance_class'])
    sn.pairplot(df)
    plt.show()


def hist(df: pd.DataFrame, col: str, discrete: bool = False):
    compare_col = 'relevance_class'
    hue_order = ['high', 'low']
    sn.histplot(data=df[[col, compare_col]], x=col, hue=compare_col, hue_order=hue_order, stat='count',
                multiple='stack', discrete=discrete)
    plt.show()


def explore_raw(query_df: pd.DataFrame, product_df: pd.DataFrame):
    # print(query_df.info())
    # print(product_df.info())
    #
    # # Venn
    # qs = set(query_df['product_uid'])
    # ps = set(product_df['product_uid'])
    # venn2([qs, ps], ('Query  {Q}', ' Description  {D}'))
    # venn2_circles([qs, ps], linestyle='dashed')
    # plt.show()
    #
    # # Score distribution
    # sn.histplot(x='relevance', data=query_df)
    # plt.show()

    # Product id
    # print('\n Products')
    # print('unique products', len(query_df['product_uid'].unique()))
    # cs = query_df['product_uid'].value_counts().reset_index()
    # cs = cs.reset_index()
    # sn.lineplot(x='index', y='count', data=cs)
    # plt.xlabel('Product')
    # plt.show()
    # print('more occurrences', len(cs[cs['count'] > 1]))
    # print('median', cs['count'].median())
    # print('mean', cs['count'].mean())

    # Search term
    # print('\n Queries')
    # sn.histplot(query_df, x='term_count', discrete=True)
    # plt.xlabel('Amount of words in search term')
    # plt.show()
    # qs = query_df['search_term'].value_counts().reset_index()
    # qs = qs.reset_index()
    # sn.lineplot(x='index', y='count', data=qs)
    # plt.xlabel('Query')
    # plt.show()
    # print('max', query_df['term_count'].max())
    # print('median', query_df['term_count'].median())
    # print('mean', query_df['term_count'].mean())
    # print('unique queries', len(query_df['search_term'].unique()))
    # print('more occurrences', len(qs[qs['count'] > 1]))
    # print('occurrence mean', qs['count'].mean())

    # word occurrence
    # ws = query_df['search_term'].apply(lambda x: x.split()).explode()
    # ws = ws.value_counts().reset_index()
    # print(ws)
    # sn.barplot(data=ws.head(20), x='count', y='search_term')
    # plt.show()
    #
    # # unit occurrence
    # query_df['length'] = query_df['search_term'].apply(len)
    # print('no words', query_df[query_df['length'] == 1])
    # print(query_df.iloc[2924])

    # # relevance
    # sn.histplot(data=query_df, x='relevance')
    # plt.show()
    #
    # uns = query_df['relevance'].value_counts().reset_index().sort_values(by='relevance')
    # print(len(uns))
    # print(uns)

    # title
    df = product_df.copy()
    print(df.info())
    print('unique stcs', len(df['product_description'].unique()))

    df['words'] = df['product_description'].apply(lambda x: x.split())
    df['word_count'] = df['words'].apply(len)
    print('mean', df['word_count'].mean())
    print('empty', len(df[df['word_count'] == 0]))

    sn.histplot(data=df, x='word_count')
    plt.xlabel('Amount of words in product description')
    plt.show()

    ws = df['words'].explode().value_counts().reset_index()
    print(ws)
    sn.barplot(data=ws.head(20), x='count', y='words')
    plt.show()
    print('total', ws['count'].sum())
    print('total', ws['count'].head(4).sum())



def explore_train(df: pd.DataFrame):
    print(df.info())

    print(df[df['id'] == 272]['numbers_in_common_query_title'])

    mask = df.columns.tolist()[19:]
    # mask = ['count_in_common_query_title', 'max_match_query_title', 'min_match_query_title']
    df = df[mask + ['relevance']]
    c = df.corr()['relevance'][:].reset_index().sort_values(by='relevance', ascending=False)
    print(c)

    cormat = df.corr()
    sn.heatmap(cormat, annot=True, annot_kws={"size": 4})
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.tight_layout()
    plt.show()


def explore():
    query_df, product_df, train_df = read_data()
    # explore_raw(query_df, product_df)
    explore_train(train_df)


if __name__ == '__main__':
    explore()
