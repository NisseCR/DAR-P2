import os.path
from typing import Callable

import pandas as pd
import numpy as np
from numpy import ndarray
from sklearn.metrics import jaccard_score
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from sklearn.metrics.pairwise import cosine_similarity
import math

# <editor-fold desc="Setup">
# Embedding files
GLOVE_FILE = "./embeddings/glove.6B.50d.txt"  # Path to the GloVe embeddings file
W2V_FILE = "./embeddings/glove.word2vec.50d.txt"  # Path to the w2v embeddings file

if not os.path.exists(W2V_FILE):
    glove2word2vec(GLOVE_FILE, W2V_FILE)

# Word embeddings
WORD_VECTORS = KeyedVectors.load_word2vec_format(W2V_FILE, binary=False)
# </editor-fold>

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


def words_in_common(query: list[str], document: list[str]) -> set[str]:
    return set(query).intersection(document)


def add_words_in_common(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df[f'words_in_common_query_{col}'] = df.apply(lambda r: words_in_common(r['query'], r[col]), axis=1)
    return df


def add_word_count(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df[f'word_count_{col}'] = df[col].apply(len)
    return df


def add_char_count(df: pd.DataFrame, col: str) -> pd.DataFrame:
    def char_count(tokens: list[str]) -> int:
        return sum([len(word) for word in tokens])

    df[f'char_count_{col}'] = df[col].apply(char_count)
    return df


def add_avg_char_count(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df[f'avg_char_count_{col}'] = df[f'char_count_{col}'] / df[f'word_count_{col}']
    return df


def add_count_in_common(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df[f'count_in_common_query_{col}'] = df[f'words_in_common_query_{col}'].apply(len)
    return df


def add_count_not_in_common(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df[f'count_not_in_common_query_{col}'] = df['word_count_query'] - df[f'count_in_common_query_{col}']
    return df


def add_all_words_in_common(df: pd.DataFrame, col: str) -> pd.DataFrame:
    new_col = f'all_words_in_common_query_{col}'
    df[new_col] = df[f'count_in_common_query_{col}'] == df[f'word_count_query']
    df[new_col] = df[new_col].replace({True: 1, False: 0})
    return df


def add_ratio_words_in_common(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df[f'ratio_words_in_common_query_{col}'] = df[f'count_in_common_query_{col}'] / df[f'word_count_query']
    df[f'ratio_words_in_common_query_{col}'] = df[f'ratio_words_in_common_query_{col}'].fillna(0)
    return df


def add_numbers_in_common(df: pd.DataFrame, col: str) -> pd.DataFrame:
    def numbers_in_common(query: list[int], document: list[int]) -> int:
        return len(set(query).intersection(document))

    df[f'numbers_in_common_query_{col}'] = df.apply(
        lambda r: numbers_in_common(r['query_numbers'], r[f'{col}_numbers']), axis=1)
    return df


def add_jaccard(df: pd.DataFrame, col: str) -> pd.DataFrame:
    def jaccard(query: list[str], document: list[str]) -> float:
        q_set = set(query)
        d_set = set(document)
        return len(q_set.intersection(d_set)) / len(q_set.union(d_set))

    df[f'jaccard_query_{col}'] = df.apply(lambda r: jaccard(r['query'], r[col]), axis=1)
    return df


def get_vector_similarity(x, y):
    return cosine_similarity(x.reshape(1, -1), y.reshape(1, -1))[0][0]


def add_document_vectors(df: pd.DataFrame, col: str) -> pd.DataFrame:
    def get_document_vector(model: KeyedVectors, tokens: list[str]) -> ndarray:
        words = [word for word in tokens if word in model]

        if len(words) == 0:
            return np.empty(0)

        return model[words]

    df[f'{col}_vs'] = df[f'{col}_std'].apply(lambda d: get_document_vector(WORD_VECTORS, d))
    return df


def add_min_cos_sims(df: pd.DataFrame, col: str) -> pd.DataFrame:
    def min_cos_sim(query_vs: ndarray, document_vs: ndarray) -> float:
        sims = [0]
        for query_v in query_vs:
            for document_v in document_vs:
                sims.append(get_vector_similarity(query_v, document_v))

        return min(sims)

    df['min_cos_sim'] = df.apply(lambda r: min_cos_sim(r['query_vs'], r[f'{col}_vs']), axis=1)
    return df


def add_cos_sim(df: pd.DataFrame, col: str) -> pd.DataFrame:
    def cos_sim(query_vs: ndarray, document_vs: ndarray) -> float:
        if not query_vs.any() or not document_vs.any():
            return 0

        query_v = np.mean(query_vs, axis=0)
        document_v = np.mean(document_vs, axis=0)
        return get_vector_similarity(query_v, document_v)

    df[f'glove_cos_sim_query_{col}'] = df.apply(lambda r: cos_sim(r['query_vs'], r[f'{col}_vs']), axis=1)
    return df


def get_unique_documents(df: pd.DataFrame, col: str) -> pd.DataFrame:
    return df[['product_uid', col]].groupby('product_uid').agg(doc=(col, 'first'))


def get_df_scores(df: pd.DataFrame) -> dict[str, int]:
    DFs = {}
    for _, row in df.iterrows():
        for word in set(row['doc']):
            if word in DFs:
                DFs[word] += 1
            else:
                DFs[word] = 1

    return DFs


def get_idf_scores(df: pd.DataFrame, col: str) -> dict[str, float]:
    df = df.copy()
    df = get_unique_documents(df, col)
    DFs = get_df_scores(df)
    N = len(df)
    return {word: math.log2(N / DF) for word, DF in DFs.items()}


def add_tf_idf(df: pd.DataFrame, col: str, IDFs: dict[str, float]) -> pd.DataFrame:
    def tf_idf_score(in_common: set[str], document: list[str], IDFs: dict[str, float]) -> float:
        tf_idf = 0

        for word in in_common:
            tf = document.count(word)
            idf = IDFs[word]
            tf_idf += tf * idf

        return tf_idf

    df[f'tf_idf_query_{col}'] = df.apply(lambda r: tf_idf_score(r[f'words_in_common_query_{col}'], r[col], IDFs), axis=1)
    return df


def add_f_length_match(df: pd.DataFrame, col: str, f: Callable, new_col: str) -> pd.DataFrame:
    def longest_match(in_common: set[str]) -> int:
        if len(in_common) == 0:
            return 0

        return len(f(in_common, key=len))

    df[f'{new_col}_match_query_{col}'] = df[f'words_in_common_query_{col}'].apply(longest_match)
    return df


def okapiBM25(d: list[str], q: list[str], idf: dict[str, int], avg_dl: float):
    return sum([okapiSingleScore(query, d, idf, avg_dl) for query in q])


def okapiSingleScore(q: str, d: list[str], idf: dict[str, int], avg_dl: float):
    f = d.count(q)
    k_1 = 1.6
    b = 0.75
    not_found_value = 11.5
    if q in idf:
        idf_ = idf[q]
    else:
        idf_ = not_found_value
    return idf_*(f*(k_1+1))/(f+k_1*(1-b+b*len(d)/avg_dl))


def classify_target(df: pd.DataFrame) -> pd.DataFrame:
    mean = df['relevance'].mean()
    df['relevance_class'] = np.where(df['relevance'] >= mean, 'high', 'low')
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    # Words in common
    df = add_words_in_common(df, 'title')
    df = add_words_in_common(df, 'description')

    # Word count
    print('Word count')
    df = add_word_count(df, 'query')
    df = add_word_count(df, 'title')
    df = add_word_count(df, 'description')

    # Character count
    print('Character count')
    df = add_char_count(df, 'query')
    df = add_char_count(df, 'title')
    df = add_char_count(df, 'description')
    df = add_avg_char_count(df, 'query')
    df = add_avg_char_count(df, 'title')
    df = add_avg_char_count(df, 'description')

    # Word comparison
    print('Word comparison')
    df = add_count_in_common(df, 'title')
    df = add_count_in_common(df, 'description')
    df = add_count_not_in_common(df, 'title')
    df = add_count_not_in_common(df, 'description')
    df = add_ratio_words_in_common(df, 'title')
    df = add_ratio_words_in_common(df, 'description')
    df = add_all_words_in_common(df, 'title')
    df = add_all_words_in_common(df, 'description')

    # Number / unit comparison
    print('Number comparison')
    df = add_numbers_in_common(df, 'title')
    df = add_numbers_in_common(df, 'description')

    # Word vectors
    print('Word vectors')
    df = add_document_vectors(df, 'query')
    df = add_document_vectors(df, 'title')
    df = add_document_vectors(df, 'description')
    df = add_cos_sim(df, 'title')
    df = add_cos_sim(df, 'description')

    # TF IDF scores
    print('TF IDF scores')
    IDFs_title = get_idf_scores(df, 'title')
    df = add_tf_idf(df, 'title', IDFs_title)
    IDFs_description = get_idf_scores(df, 'description')
    df = add_tf_idf(df, 'description', IDFs_description)

    # Length of match
    df = add_f_length_match(df, 'title', max, 'max')
    df = add_f_length_match(df, 'description', max, 'max')
    df = add_f_length_match(df, 'title', min, 'min')
    df = add_f_length_match(df, 'description', min, 'min')
    return df

