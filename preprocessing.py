from typing import Tuple, Any
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.stem import PorterStemmer
import re


# Setup
nltk.download('stopwords')
nltk.download('punkt')
STOP_WORDS = set(stopwords.words())
PS = PorterStemmer()


def read_data() -> pd.DataFrame:
    query_df = pd.read_csv('./data/query_product.csv', encoding='latin1')
    product_df = pd.read_csv('./data/product_descriptions.csv', encoding='latin1')
    df = pd.merge(query_df, product_df, on='product_uid', how='left')
    return df


def remove_punctuation(sentence: str) -> str:
    return re.sub(r'[^\w\s]', ' ', sentence)


def remove_numbers(sentence: str) -> str:
    return re.sub(r'[0-9]', ' ', sentence)


def remove_stopwords(tokens: list[str]) -> list[str]:
    return [word for word in tokens if word not in STOP_WORDS]


def stemming(tokens: list[str]) -> list[str]:
    return [PS.stem(word) for word in tokens]


def sentence_pipeline(sentence: str) -> str:
    sentence = sentence.lower()
    sentence = remove_punctuation(sentence)
    sentence = remove_numbers(sentence)
    return sentence


def tokenize_pipeline(tokens: list[str]) -> list[str]:
    tokens = remove_stopwords(tokens)
    tokens = stemming(tokens)
    return tokens


def npl_pipeline(sentence: str) -> str:
    sentence = sentence_pipeline(sentence)
    tokens = word_tokenize(sentence)
    tokens = tokenize_pipeline(tokens)
    sentence = TreebankWordDetokenizer().detokenize(tokens)
    return sentence


def preprocess():
    # df = read_data()

    test = "#123 The Big programmer anti-ui; stuff, packaged?!I'm doing a tested test at the moment."

    print(test)
    test = npl_pipeline(test)
    print(test)


if __name__ == '__main__':
    preprocess()
