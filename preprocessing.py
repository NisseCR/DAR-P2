import pandas as pd

from models.features import add_features, classify_target
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


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={'product_title': 'title', 'product_description': 'description', 'search_term': 'query'})


def add_backup(df: pd.DataFrame) -> pd.DataFrame:
    # Save original data
    df['query_org'] = df['query']
    df['title_org'] = df['title']
    df['description_org'] = df['description']
    return df


def tokenize_data(df: pd.DataFrame) -> pd.DataFrame:
    df = tokenize(df, 'title')
    df = tokenize(df, 'query')
    return df


def preprocessing_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    df = rename_columns(df)
    df = add_backup(df)
    df = tokenize_data(df)
    df = classify_target(df)
    df = add_features(df)
    return df


def export(df: pd.DataFrame):
    df.to_csv('./data/data.csv')


def preprocess(doc_name: str, top: int = None):
    # Read
    df = read_data(top)

    # Preprocess
    df = preprocessing_pipeline(df)

    # Exporting
    export(df)

    # Debug
    print(df.columns)
    print(df.head(10))


if __name__ == '__main__':
    preprocess('product_title', 20)

