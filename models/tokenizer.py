import string
from typing import Set, Any

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import re


# <editor-fold desc="Setup">
# NLTK downloads
nltk.download('stopwords')
nltk.download('punkt')

# Data cleaning constants
PS = PorterStemmer()
WL = WordNetLemmatizer()
STOP_LIST = set(stopwords.words() + [*string.punctuation])
# </editor-fold>


def isolate_numbers(sentence: str) -> set[str]:
    return set(re.findall(r'\d+', sentence))


def tokenize_sentence(sentence: str) -> list[str]:
    """
    Apply text preprocessing to the given sentence:
    - Remove punctuation
    - Remove stopwords
    - Remove numbers
    - Transform to lowercase

    TODO spelling check
    :param sentence: raw sentence
    :return: tokenized sentence
    """
    return [
        word
        for word in word_tokenize(sentence.lower())
        if word not in STOP_LIST
    ]


def stem_tokens(tokens: list[str]) -> list[str]:
    return [PS.stem(word) for word in tokens]


def tokenize(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df[f'{col}_numbers'] = df[col].apply(isolate_numbers)
    df[f'{col}_non_stem'] = df[col].apply(tokenize_sentence)
    df[f'{col}'] = df[f'{col}_non_stem'].apply(stem_tokens)
    return df

