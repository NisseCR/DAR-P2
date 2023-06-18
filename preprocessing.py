import pandas as pd
from sklearn.model_selection import train_test_split

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
    df = tokenize(df, 'query')
    df = tokenize(df, 'title')
    df = tokenize(df, 'description')
    return df


def preprocessing_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    print('Rename columns')
    df = rename_columns(df)

    print('Add backup')
    df = add_backup(df)

    print('Tokenize data')
    df = tokenize_data(df)

    print('Add features')
    df = classify_target(df)
    df = add_features(df)
    return df


def export(df: pd.DataFrame, set_type: str):
    df.to_csv(f'./data/{set_type}.csv')


def preprocess_set(df: pd.DataFrame, set_type: str):
    print(f'Preprocess {set_type}')
    # Preprocess
    df = preprocessing_pipeline(df)

    # Exporting
    export(df, set_type)

    # Debug
    print(df.columns)
    print(df.head(10))


def preprocess(top: int = None):
    # Read
    df = read_data(top)

    # Create train and test sets
    if top is None:
        training_size = 50000
    else:
        training_size = round(top * (3 / 4))

    train_df, test_df = train_test_split(df, test_size=(len(df) - training_size), random_state=42)

    # Preprocess
    preprocess_set(train_df, 'train')
    preprocess_set(test_df, 'test')


if __name__ == '__main__':
    preprocess(top=None)

