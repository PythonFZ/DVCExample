import os
import pickle
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sparse
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from zntrack import nodify, NodeConfig


@nodify(
    params={"max_features": 500, "ngrams": 1},
    deps=[Path("data", "prepared"), Path("src", "featurization.py")],
    outs=Path("data", "features"),
)
def featurize(cfg: NodeConfig):
    np.set_printoptions(suppress=True)
    train_input = cfg.deps[0] / "train.tsv"
    test_input = cfg.deps[0] / "test.tsv"
    train_output = cfg.outs / "train.pkl"
    test_output = cfg.outs / "test.pkl"

    cfg.outs.mkdir(exist_ok=True, parents=True)

    # Generate train feature matrix
    df_train = get_df(train_input)
    train_words = np.array(df_train.text.str.lower().values.astype("U"))

    bag_of_words = CountVectorizer(
        stop_words="english",
        max_features=cfg.params.max_features,
        ngram_range=(1, cfg.params.ngrams),
    )

    bag_of_words.fit(train_words)
    train_words_binary_matrix = bag_of_words.transform(train_words)
    tfidf = TfidfTransformer(smooth_idf=False)
    tfidf.fit(train_words_binary_matrix)
    train_words_tfidf_matrix = tfidf.transform(train_words_binary_matrix)

    save_matrix(df_train, train_words_tfidf_matrix, train_output)

    # Generate test feature matrix
    df_test = get_df(test_input)
    test_words = np.array(df_test.text.str.lower().values.astype("U"))
    test_words_binary_matrix = bag_of_words.transform(test_words)
    test_words_tfidf_matrix = tfidf.transform(test_words_binary_matrix)

    save_matrix(df_test, test_words_tfidf_matrix, test_output)


def get_df(data):
    df = pd.read_csv(
        data,
        encoding="utf-8",
        header=None,
        delimiter="\t",
        names=["id", "label", "text"],
    )
    sys.stderr.write(f"The input data frame {data} size is {df.shape}\n")
    return df


def save_matrix(df, matrix, output):
    id_matrix = sparse.csr_matrix(df.id.astype(np.int64)).T
    label_matrix = sparse.csr_matrix(df.label.astype(np.int64)).T

    result = sparse.hstack([id_matrix, label_matrix, matrix], format="csr")

    msg = "The output matrix {} size is {} and data type is {}\n"
    sys.stderr.write(msg.format(output, result.shape, result.dtype))

    with open(output, "wb") as fd:
        pickle.dump(result, fd)
    pass
