import pandas as pd

from models.features import add_features
from models.tokenizer import tokenize

# <editor-fold desc="Setup">
# Pandas settings
pd.options.display.max_columns = 10
pd.options.display.max_colwidth = 30
# </editor-fold>


def read_data(top: int = None) -> pd.DataFrame:
    query_df = pd.read_csv('./data/query_product.csv', encoding='latin1')
    product_df = pd.read_csv('./data/product_descriptions.csv', encoding='latin1')
    df = pd.merge(query_df, product_df, on='product_uid', how='left')

    # Take top n elements when testing
    if top is not None:
        df = df.head(top)

    return df


def truncate_unusable_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=['ratio_in_common'])
    return df


def export(df: pd.DataFrame):
    df.to_csv('./data/data.csv')


def preprocess(doc_name: str):
    # Read from csv
    df = read_data()

    # Rename document column
    df = df.rename(columns={doc_name: 'doc', 'search_term': 'query'})
    df = df[['id', 'product_uid', 'query', 'doc', 'relevance']]

    # NPL preprocessing pipeline
    df = tokenize(df, 'doc')
    df = tokenize(df, 'query')

    # Feature extraction
    df = add_features(df)

    # Truncate data
    df = truncate_unusable_data(df)

    # Exporting
    export(df)
    print(df.columns)
    print(df.head(10))


if __name__ == '__main__':
    preprocess('product_title')

