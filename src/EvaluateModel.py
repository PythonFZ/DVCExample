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


class TFModel(ZnTrackOption):
    dvc_option = "outs"
    zn_type = utils.ZnTypes.RESULTS

    def get_filename(self, instance) -> pathlib.Path:
        return pathlib.Path("nodes", instance.node_name, "model")

    def save(self, instance):
        model = self.__get__(instance, self.owner)
        file = self.get_filename(instance)
        model.save(file)

    def get_data_from_files(self, instance):
        file = self.get_filename(instance)
        model = keras.models.load_model(file)
        return model


# with this custom Type we can define `model = TFModel()` and use it similar to the other `zn.<options>` but passing it a TensorFlow model.
# Note: You can also register a custom `znjson` de/serializer and use `zn.outs` instead.
# 
# In this simple example we only define the epochs as parameters. For a more advanced Node you would try to catch all parameters, such as layer types, neurons, ... as `zn.params`.

# In[28]:


class MLModel(Node):
    train_data: DataPreprocessor = zn.deps(DataPreprocessor)
    training_history = zn.plots()
    metrics = zn.metrics()

    epochs = zn.params(10)

    model = TFModel()

    def __init__(self, epochs: int = 10, **kwargs):
        super().__init__(**kwargs)
        self.epochs = epochs

        self.optimizer = "adam"

    def run(self):
        self.build_model()
        self.train_model()

    def train_model(self):
        """Train the model"""
        self.model.compile(
            optimizer=self.optimizer,
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        history = self.model.fit(
            self.train_data.train,
            self.train_data.labels,
            validation_split=0.3,
            epochs=self.epochs,
            batch_size=64,
        )
        self.training_history = pd.DataFrame(history.history)
        self.training_history.index.name = "epoch"
        # use the last values for model metrics
        self.metrics = dict(self.training_history.iloc[-1])

    def build_model(self):
        """Build the model using keras.Sequential API"""
        self.model = keras.Sequential(
            [
                layers.Conv2D(
                    filters=32,
                    kernel_size=(3, 3),
                    input_shape=(28, 28, 1),
                    activation="relu",
                    padding="same",
                ),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(
                    filters=64, kernel_size=(3, 3), padding="same", activation="relu"
                ),
                layers.MaxPooling2D(2, 2),
                layers.Conv2D(
                    64, kernel_size=(3, 3), padding="same", activation="relu"
                ),
                layers.Flatten(),
                layers.Dense(64, activation="relu"),
                layers.Dense(25, activation="softmax"),
            ]
        )


class EvaluateModel(Node):
    metrics = zn.metrics()
    ml_model: MLModel = zn.deps(MLModel)
    test_data: DataPreprocessor = zn.deps()

    def run(self):
        loss, accuracy = self.ml_model.model.evaluate(
            self.test_data.train, self.test_data.labels
        )
        self.metrics = {"loss": loss, "accuracy": accuracy}


