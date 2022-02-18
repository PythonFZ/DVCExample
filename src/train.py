import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from zntrack import Node, dvc, zn

from .featurization import Featurize


class Train(Node):
    featurize: Featurize = zn.deps(Featurize)
    seed = zn.params(20170428)
    n_est = zn.params(50)
    min_split = zn.params(2)
    model_file = dvc.outs("model.pkl")
    _self = dvc.deps(Path("src", "train.py"))

    def run(self):
        with open(self.featurize.train_output, "rb") as fd:
            matrix = pickle.load(fd)

        labels = np.squeeze(matrix[:, 1].toarray())
        x = matrix[:, 2:]

        sys.stderr.write("Input matrix size {}\n".format(matrix.shape))
        sys.stderr.write("X matrix size {}\n".format(x.shape))
        sys.stderr.write("Y matrix size {}\n".format(labels.shape))

        clf = RandomForestClassifier(
            n_estimators=self.n_est,
            min_samples_split=self.min_split,
            n_jobs=2,
            random_state=self.seed,
        )

        clf.fit(x, labels)

        with open(self.model_file, "wb") as fd:
            pickle.dump(clf, fd)
