import json
import math
import pickle
from pathlib import Path

import sklearn.metrics as metrics
from zntrack import Node, dvc, zn

from .train import Train


class Evaluate(Node):
    train: Train = zn.deps(Train)
    metrics = zn.metrics()
    prc_file = dvc.plots_no_cache("prc.json")
    roc_file = dvc.plots_no_cache("roc.json")
    _self = dvc.deps(Path("src", "evaluate.py"))

    def run(self):
        with open(self.train.model_file, "rb") as fd:
            model = pickle.load(fd)

        with open(self.train.featurize.test_output, "rb") as fd:
            matrix = pickle.load(fd)

        labels = matrix[:, 1].toarray()
        x = matrix[:, 2:]

        predictions_by_class = model.predict_proba(x)
        predictions = predictions_by_class[:, 1]

        precision, recall, prc_thresholds = metrics.precision_recall_curve(
            labels, predictions
        )
        fpr, tpr, roc_thresholds = metrics.roc_curve(labels, predictions)

        avg_prec = metrics.average_precision_score(labels, predictions)
        roc_auc = metrics.roc_auc_score(labels, predictions)

        self.metrics = {"avg_prec": avg_prec, "roc_auc": roc_auc}

        # ROC has a drop_intermediate arg that reduces the number of points.
        # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html#sklearn.metrics.roc_curve.
        # PRC lacks this arg, so we manually reduce to 1000 points as a rough estimate.
        nth_point = math.ceil(len(prc_thresholds) / 1000)
        prc_points = list(zip(precision, recall, prc_thresholds))[::nth_point]
        with open(self.prc_file, "w") as fd:
            json.dump(
                {
                    "prc": [
                        {"precision": p, "recall": r, "threshold": t}
                        for p, r, t in prc_points
                    ]
                },
                fd,
                indent=4,
            )

        with open(self.roc_file, "w") as fd:
            json.dump(
                {
                    "roc": [
                        {"fpr": fp, "tpr": tp, "threshold": t}
                        for fp, tp, t in zip(fpr, tpr, roc_thresholds)
                    ]
                },
                fd,
                indent=4,
            )
