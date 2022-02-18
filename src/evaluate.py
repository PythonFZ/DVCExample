import json
import math
import pickle

import sklearn.metrics as metrics
from pathlib import Path

from zntrack import nodify, NodeConfig


@nodify(
    params={"seed": 20170428, "n_est": 50, "min_split": 2},
    deps=[Path("data", "features"), "model.pkl", Path("src", "evaluate.py")],
    metrics_no_cache="scores.json",
    plots_no_cache=["prc.json", "roc.json"]
)
def evaluate(cfg: NodeConfig):


    model_file = cfg.deps[1]
    matrix_file = cfg.deps[0] / "test.pkl"
    scores_file = cfg.metrics_no_cache
    prc_file = cfg.plots_no_cache[0]
    roc_file = cfg.plots_no_cache[1]

    with open(model_file, "rb") as fd:
        model = pickle.load(fd)

    with open(matrix_file, "rb") as fd:
        matrix = pickle.load(fd)

    labels = matrix[:, 1].toarray()
    x = matrix[:, 2:]

    predictions_by_class = model.predict_proba(x)
    predictions = predictions_by_class[:, 1]

    precision, recall, prc_thresholds = metrics.precision_recall_curve(labels, predictions)
    fpr, tpr, roc_thresholds = metrics.roc_curve(labels, predictions)

    avg_prec = metrics.average_precision_score(labels, predictions)
    roc_auc = metrics.roc_auc_score(labels, predictions)

    with open(scores_file, "w") as fd:
        json.dump({"avg_prec": avg_prec, "roc_auc": roc_auc}, fd, indent=4)

    # ROC has a drop_intermediate arg that reduces the number of points.
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html#sklearn.metrics.roc_curve.
    # PRC lacks this arg, so we manually reduce to 1000 points as a rough estimate.
    nth_point = math.ceil(len(prc_thresholds) / 1000)
    prc_points = list(zip(precision, recall, prc_thresholds))[::nth_point]
    with open(prc_file, "w") as fd:
        json.dump(
            {
                "prc": [
                    {"precision": p, "recall": r, "threshold": t} for p, r, t in prc_points
                ]
            },
            fd,
            indent=4,
        )

    with open(roc_file, "w") as fd:
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
