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
from zntrack.core import ZnTrackOption
from zntrack import utils


@nodify(outs="dataset", params={"dataset": "datamunge/sign-language-mnist"})
def download_kaggle(cfg: NodeConfig):
    kaggle.api.dataset_download_files(
        dataset=cfg.params.dataset, path=cfg.outs, unzip=True
    )


class DataPreprocessor(Node):
    data: pathlib.Path = dvc.deps(pathlib.Path("dataset"))
    dataset = zn.params("sign_mnist_test")

    train: np.ndarray = zn.outs()
    labels: np.ndarray = zn.outs()

    def run(self):
        df = pd.read_csv((self.data / self.dataset / self.dataset).with_suffix(".csv"))

        self.labels = df.values[:, 0]
        self.labels = to_categorical(self.labels)
        self.train = df.values[:, 1:]

        self.normalize_and_scale_data()

    def normalize_and_scale_data(self):
        self.train = self.train / 255
        self.train = self.train.reshape((-1, 28, 28, 1))

    def plot_image(self, index):
        plt.imshow(self.train[index])
        plt.title(f"Label {self.labels[index].argmax()}")


