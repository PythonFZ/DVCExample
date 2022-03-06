import kaggle
from zntrack import nodify, NodeConfig, config
import matplotlib.pyplot as plt
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from tensorflow.keras.utils import to_categorical
from tensorflow import keras
from tensorflow.keras import layers
from zntrack import Node, zn, dvc
import pathlib


@nodify(outs="dataset", params={"dataset": "datamunge/sign-language-mnist"})
def download_kaggle(cfg: NodeConfig):
    kaggle.api.dataset_download_files(
        dataset=cfg.params.dataset, path=cfg.outs, unzip=True
    )


