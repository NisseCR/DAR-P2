import string
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from gensim.models import KeyedVectors
from gensim.test.utils import datapath, get_tmpfile
from gensim.scripts.glove2word2vec import glove2word2vec

# Pandas settings
pd.options.display.max_columns = 10
pd.options.display.max_colwidth = 30


# Setup
nltk.download('stopwords')
nltk.download('punkt')

PS = PorterStemmer()
STOP_LIST = set(stopwords.words() + [*string.punctuation])


# <editor-fold desc="Import data">
def read_data(top: int = None) -> pd.DataFrame:
    query_df = pd.read_csv('./data/query_product.csv', encoding='latin1')
    product_df = pd.read_csv('./data/product_descriptions.csv', encoding='latin1')
    df = pd.merge(query_df, product_df, on='product_uid', how='left')

    # Take top n elements when testing
    if top is not None:
        df = df.head(top)

    return df
# </editor-fold>


# <editor-fold desc="Text preprocessing">
def npl_pipeline(sentence: str) -> list[str]:
    """
    Apply text preprocessing to the given sentence:
    - Remove punctuation
    - Remove stopwords
    - Remove numbers
    - Transform to lowercase
    - Apply stemming
    :param sentence: raw sentence
    :return: tokenized sentence
    """
    return [
        PS.stem(word)
        for word in word_tokenize(sentence.lower())
        if word not in STOP_LIST
        and not word.isdigit()
    ]


def data_preprocessing(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df[col] = df[col].apply(npl_pipeline)
    return df
# </editor-fold>


# <editor-fold desc="Add regression features">
# possible features to add/explore:
# [ ] measurement shit -> 100feet is similar to 120ft
# [ ] colour similarity (nicks idee)
# [ ] word count
# [ ] avg word len search
# [ ] number of characters
# [ ] jaccard between search and document
# [ ] cosin coefficient between search and document (zie site van nick)
# [ ] Last word in query in document field (example: red power drill, drill is most important term)
# [ ] Vector space model shit?
# [ ] Language model -> dirichlet, absolute and jelinek miller something
# [ ] okapi BM25
# [ ] Mischien words in common vervangen met tfidf
# [ ] (sum, min, max) of (tf, idf, tf-idf) for the search query in each of the text field (zie site)



def words_in_common(query: list[str], doc: list[str]) -> int:
    return len(set(query).intersection(doc))


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df['len_of_query'] = df['query'].apply(len)
    df['len_of_doc'] = df['doc'].apply(len)
    df['words_in_common'] = df.apply(lambda r: words_in_common(r['query'], r['doc']), axis=1)
    df['ratio_in_common'] = df['words_in_common'] / df['len_of_query']
    df['complete_ratio'] = df['ratio_in_common'] == 1
    return df
# </editor-fold>


# <editor-fold desc="Truncate data">
def truncate_unusable_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=['ratio_in_common'])
    return df
# </editor-fold>


# <editor-fold desc="Export data">
def export(df: pd.DataFrame):
    df.to_csv('./data/data.csv')
# </editor-fold>


def preprocess(doc_name: str):
    # Read from csv
    df = read_data(20)

    # Rename document column
    df = df.rename(columns={doc_name: 'doc', 'search_term': 'query'})
    df = df[['id', 'product_uid', 'query', 'doc', 'relevance']]

    # NPL preprocessing pipeline
    df = data_preprocessing(df, 'doc')
    df = data_preprocessing(df, 'query')

    # Feature extraction
    df = add_features(df)

    # Truncate data
    df = truncate_unusable_data(df)

    # Exporting
    export(df)
    print(df.columns)
    print(df.head(10))


def test():
    # Test
    data = "#123 The Big programmer anti-ui; stuff, packaged?!I'm doing a tested test at the moment."

    print(data)
    data = npl_pipeline(data)
    print(data)


if __name__ == '__main__':
    preprocess('product_title')
