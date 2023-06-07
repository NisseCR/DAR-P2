from typing import Tuple, Any
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer


# Setup
nltk.download()
STOP_WORDS = set(stopwords.words('punkt'))


def read_data() -> pd.DataFrame:
    query_df = pd.read_csv('./data/query_product.csv', encoding='latin1')
    product_df = pd.read_csv('./data/product_descriptions.csv', encoding='latin1')
    df = pd.merge(query_df, product_df, on='product_uid', how='left')
    return df


def remove_stopwords(tokens: list[str]) -> list[str]:
    return [word for word in tokens if word not in STOP_WORDS]


def tokenize_pipeline(sentence: str) -> str:
    tokens = word_tokenize(sentence)
    tokens = remove_stopwords(tokens)
    sentence = TreebankWordDetokenizer().detokenize(tokens)
    return sentence


def preprocess():
    df = read_data()

    test = "im a and tower big test in the"

    print(test)
    test = tokenize_pipeline(test)
    print(test)


if __name__ == '__main__':
    preprocess()
