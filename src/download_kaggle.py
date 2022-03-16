from zntrack import config
import pathlib
import kaggle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from zntrack import Node, NodeConfig, dvc, nodify, utils, zn
from zntrack.core import ZnTrackOption


@nodify(
    outs="dataset",
    params={"dataset": "datamunge/sign-language-mnist"}
)
def download_kaggle(cfg: NodeConfig):
    """Download dataset from kaggle"""
    kaggle.api.dataset_download_files(
        dataset=cfg.params.dataset, path=cfg.outs, unzip=True
    )


