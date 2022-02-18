import os
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from zntrack import NodeConfig, nodify


@nodify(
    params={"seed": 20170428, "n_est": 50, "min_split": 2},
    deps=[Path("data", "features"), Path("src", "train.py")],
    outs="model.pkl",
)
def train(cfg: NodeConfig):

    with open(os.path.join(cfg.deps[0], "train.pkl"), "rb") as fd:
        matrix = pickle.load(fd)

    labels = np.squeeze(matrix[:, 1].toarray())
    x = matrix[:, 2:]

    sys.stderr.write("Input matrix size {}\n".format(matrix.shape))
    sys.stderr.write("X matrix size {}\n".format(x.shape))
    sys.stderr.write("Y matrix size {}\n".format(labels.shape))

    clf = RandomForestClassifier(
        n_estimators=cfg.params.n_est,
        min_samples_split=cfg.params.min_split,
        n_jobs=2,
        random_state=cfg.params.seed,
    )

    clf.fit(x, labels)

    with open(cfg.outs, "wb") as fd:
        pickle.dump(clf, fd)
