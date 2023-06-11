import string
from typing import Tuple, Any, List
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.stem import PorterStemmer
import re


# Setup
nltk.download('stopwords')
STOP_LIST = set(stopwords.words() + [*string.punctuation])
PS = PorterStemmer()


def read_data() -> pd.DataFrame:
    query_df = pd.read_csv('./data/query_product.csv', encoding='latin1')
    product_df = pd.read_csv('./data/product_descriptions.csv', encoding='latin1')
    df = pd.merge(query_df, product_df, on='product_uid', how='left')
    return df


def npl_pipeline(sentence: str) -> list[str]:
    return [
        PS.stem(word.lower())
        for word in word_tokenize(sentence)
        if word.lower() not in STOP_LIST and not word.isdigit()
    ]


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df['product_title'] = df['product_title'].apply(npl_pipeline)
    # df['product_description'] = df['product_description'].apply(npl_pipeline)
    return df


def export(df: pd.DataFrame):
    df.to_csv('./data/data.csv')


def preprocess():
    df = read_data()
    df = clean_data(df)
    # export(df)

    # Test
    test = "#123 The Big programmer anti-ui; stuff, packaged?!I'm doing a tested test at the moment."

    print(test)
    test = npl_pipeline(test)
    print(test)


if __name__ == '__main__':
    preprocess()
