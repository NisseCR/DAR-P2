import os.path
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


def words_in_common(query: list[str], document: list[str]) -> set[str]:
    return set(query).intersection(document)


def add_words_in_common(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df[f'words_in_common_query_{col}'] = df.apply(lambda r: len(words_in_common(r['query'], r[col])), axis=1)
    return df


def add_ratio_words_in_common(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df[f'ratio_words_in_common_query_{col}'] = df[f'words_in_common_query_{col}'] / df[f'word_count_query']
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


def get_document_vector(model: KeyedVectors, tokens: list[str]) -> ndarray | None:
    words = [word for word in tokens if word in model]

    if len(words) == 0:
        return None

    vectors = model[words]
    return np.concatenate([
        np.mean(vectors, axis=0),
        np.amax(vectors, axis=0)
    ])


def add_cos_sim(df: pd.DataFrame, col: str) -> pd.DataFrame:
    def cos_sim(query: list[str], document: list[str]):
        qv = get_document_vector(WORD_VECTORS, query)
        dv = get_document_vector(WORD_VECTORS, document)

        if qv is None or dv is None:
            return 0

        return get_vector_similarity(qv, dv)

    df['glove_cos_sim'] = df.apply(lambda r: cos_sim(r['query_std'], r[f'{col}_std']), axis=1)
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
    def get_tf_idf_score(query: list[str], document: list[str], IDFs: dict[str, float]) -> float:
        in_common = words_in_common(query, document)

        tf_idf = 0

        for word in in_common:
            tf = document.count(word)
            idf = IDFs[word]
            tf_idf += tf * idf

        return tf_idf

    df[f'tf_idf_query_{col}'] = df.apply(lambda r: get_tf_idf_score(r['query'], r[col], IDFs), axis=1)
    return df


def okapiBM25(d: list[str],q: list[str], idf:dict[str, int], avg_dl: float):
    return sum([okapiSingleScore(query, d, idf, avg_dl) for query in q])


def okapiSingleScore(q: str, d:list[str], idf:dict[str, int], avg_dl: float):
    f = d.count(q)
    k_1 = 1.6
    b = 0.75
    not_found_value = 11.5
    if q in idf:
        idf_ = idf[q]
    else:
        idf_ = not_found_value
    return idf_*(f*(k_1+1))/(f+k_1*(1-b+b*len(d)/avg_dl))


def tf_idf(q: list[str], d: list[str], idf: dict[str, int]):
    res = 0
    for q_word in q:
        tf = 0
        if q_word in idf:
            idf_ = idf[q_word]
        else:
            idf_ = 11.5 #magic numberraosdruasdfjasdf
        for d_word in d:
            if d_word == q_word:
                tf += 1
        res += tf*idf_
    return res


def classify_target(df: pd.DataFrame) -> pd.DataFrame:
    mean = df['relevance'].mean()
    df['relevance_class'] = np.where(df['relevance'] >= mean, 'high', 'low')
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
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
    df = add_words_in_common(df, 'title')
    df = add_words_in_common(df, 'description')
    df = add_ratio_words_in_common(df, 'title')
    df = add_ratio_words_in_common(df, 'description')

    # Number / unit comparison
    print('Number comparison')
    df = add_numbers_in_common(df, 'title')
    df = add_numbers_in_common(df, 'description')

    # Word vectors
    print('Word vectors')
    df = add_cos_sim(df, 'title')

    # TF IDF scores
    print('TF IDF scores')
    IDFs_description = get_idf_scores(df, 'description')
    df = add_tf_idf(df, 'description', IDFs_description)

    # Eventual features
    df = df[[
        'relevance',
        'word_count_query',
        'ratio_words_in_common_query_title',
        'ratio_words_in_common_query_description',
        'numbers_in_common_query_title',
        'glove_cos_sim',
        'tf_idf_query_description'
    ]]
    return df


def _add_features(df: pd.DataFrame) -> pd.DataFrame:
    # Basic features
    df2 = pd.unique(df['product_uid'])
    df3 = pd.DataFrame({'product_uid':df2})
    unique_products = pd.merge(df3, df, on='product_uid', how='inner')
    idf = calculate_idf(unique_products)
    avg_doc_len = sum([len(row['doc']) for _,row in unique_products.iterrows()]) #vervang dit, niet zo efficent
    df['okapiBM25'] = df.apply(lambda r: okapiBM25(r['doc_stem'], r['query_stem'], idf, avg_doc_len), axis=1)
    df['tf-idf'] = df.apply(lambda r: tf_idf(r['query_stem'], r['doc_stem'], idf), axis=1)
    # Embedding metrics

    # df['query_vector'] = df['query'].apply(lambda s: get_sentence_vector(WORD_VECTORS, s))
    # df['doc_vector'] = df['doc'].apply(lambda s: get_sentence_vector(WORD_VECTORS, s))
    # df['sentence_cosine_similarity'] = df.apply(lambda r: get_vector_similarity(r['query_vector'], r['doc_vector']), axis=1)
    # df['word_count'] = df.apply(lambda r: word_count(r['doc']), axis=1)
    # df['char_count'] = df.apply(lambda r: char_count(r['doc']), axis=1)
    # df['avg_char_count'] = df.apply(lambda r: avg_char_count(r['doc']), axis=1)
    # df['jac'] = df.apply(lambda r: jac(r['query'], r['doc']), axis=1)
    return df

