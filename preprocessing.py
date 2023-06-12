import string
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import jaccard_score
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from sklearn.metrics.pairwise import cosine_similarity

# <editor-fold desc="Setup">
# Pandas settings
pd.options.display.max_columns = 10
pd.options.display.max_colwidth = 30


# NLTK downloads
nltk.download('stopwords')
nltk.download('punkt')

# Data cleaning constants
PS = PorterStemmer()
WL = WordNetLemmatizer()
STOP_LIST = set(stopwords.words() + [*string.punctuation])

# Embedding files
GLOVE_FILE = "./embeddings/glove.6B.50d.txt"  # Path to the GloVe embeddings file
W2V_FILE = "./embeddings/glove.word2vec.50d.txt"  # Path to the w2v embeddings file
_ = glove2word2vec(GLOVE_FILE, W2V_FILE)

# Word embeddings
WORD_VECTORS = KeyedVectors.load_word2vec_format(W2V_FILE, binary=False)
# </editor-fold>


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
        WL.lemmatize(word)
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
# [ ] avg word length in search
# [ ] number of characters
# [ ] jaccard between search and document
# [ ] cosin coefficient between search and document (zie site van nick)
# [ ] Last word in query in document field (example: red power drill, drill is most important term)
# [ ] Vector space model shit?
# [ ] Language model -> dirichlet, absolute and jelinek miller something
# [ ] okapi BM25
# [ ] Mischien words in common vervangen met tfidf
# [ ] (sum, min, max) of (tf, idf, tf-idf) for the search query in each of the text field (zie site)

def word_count(l: list[str]) -> int:
    return len(l)


def char_count(l : list[str]) -> int:
    return sum([len(i) for i in l])


def avg_char_count(l : list[str]) -> float:
    return char_count(l)/word_count(l)


def jac(query: list[str], doc: list[str]) -> float:
    q_set = set(query)
    d_set = set(doc)
    return len(q_set.intersection(d_set))/len(q_set.union(d_set))


def words_in_common(query: list[str], doc: list[str]) -> int:
    return len(set(query).intersection(doc))


def get_sentence_vector(model: KeyedVectors, words: list[str]):
    words = [word for word in words if word in model]
    if len(words) >= 1:
        return np.mean(model[words], axis=0)
    else:
        return []


def get_vector_similarity(x, y):
    return cosine_similarity(x.reshape(1, -1), y.reshape(1, -1))[0][0]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df['len_of_query'] = df['query'].apply(len)
    df['len_of_doc'] = df['doc'].apply(len)
    df['words_in_common'] = df.apply(lambda r: words_in_common(r['query'], r['doc']), axis=1)
    df['ratio_in_common'] = df['words_in_common'] / df['len_of_query']
    df['complete_ratio'] = df['ratio_in_common'] == 1

    # Embedding metrics
    # df['query_vector'] = df['query'].apply(lambda s: get_sentence_vector(WORD_VECTORS, s))
    # df['doc_vector'] = df['doc'].apply(lambda s: get_sentence_vector(WORD_VECTORS, s))
    # df['sentence_cosine_similarity'] = df.apply(lambda r: get_vector_similarity(r['query_vector'], r['doc_vector']), axis=1)
    df['word_count'] = df.apply(lambda r: word_count(r['doc']), axis=1)
    df['char_count'] = df.apply(lambda r: char_count(r['doc']), axis=1)
    df['avg_char_count'] = df.apply(lambda r: avg_char_count(r['doc']), axis=1)
    df['jac'] = df.apply(lambda r: jac(r['query'], r['doc']), axis=1)
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
    df = read_data()

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
    s1 = "#123 The Big programmer anti-ui; stuff, packaged?!I'm doing a tested test at the moment."
    s1 = npl_pipeline(s1)

    s2 = "Hello there, i am general grevious and i have programmed the matrix moment in this package"
    s2 = npl_pipeline(s2)

    s3 = "kitty pink learns skipping mountains"
    s3 = npl_pipeline(s3)

    print(s1)
    print(s2)
    print(s3)

    x = get_sentence_vector(WORD_VECTORS, s1)
    y = get_sentence_vector(WORD_VECTORS, s2)
    z = get_sentence_vector(WORD_VECTORS, s3)

    r = get_vector_similarity(x, y)
    print(r)

    r = get_vector_similarity(x, z)
    print(r)


if __name__ == '__main__':
    preprocess('product_title')
    test()

