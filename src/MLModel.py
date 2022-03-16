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


class DataPreprocessor(Node):
    """Prepare kaggle dataset for training

    * normalize and reshape the features
    * one-hot encode the labels
    """
    # dependecies and parameters
    data: pathlib.Path = dvc.deps(pathlib.Path("dataset"))
    dataset = zn.params("sign_mnist_test")
    # outputs
    features: np.ndarray = zn.outs()
    labels: np.ndarray = zn.outs()

    def run(self):
        """Primary Node Method"""
        df = pd.read_csv((self.data / self.dataset / self.dataset).with_suffix(".csv"))

        self.labels = df.values[:, 0]
        self.labels = to_categorical(self.labels)
        self.features = df.values[:, 1:]

        self.normalize_and_scale_data()

    def normalize_and_scale_data(self):
        self.features = self.features / 255
        self.features = self.features.reshape((-1, 28, 28, 1))

    def plot_image(self, index):
        plt.imshow(self.features[index])
        plt.title(f"Label {self.labels[index].argmax()}")


class TFModel(ZnTrackOption):
    dvc_option = "outs"
    zn_type = utils.ZnTypes.RESULTS

    def get_filename(self, instance) -> pathlib.Path:
        """Filename depending on the instance node_name"""
        return pathlib.Path("nodes", instance.node_name, "model")

    def save(self, instance):
        """Serialize and save values to file"""
        model = self.__get__(instance, self.owner)
        file = self.get_filename(instance)
        model.save(file)

    def get_data_from_files(self, instance):
        """Load values from file and deserialize"""
        file = self.get_filename(instance)
        model = keras.models.load_model(file)
        return model


# with this custom Type we can define `model = TFModel()` and use it similar to the other `zn.<options>` but passing it a TensorFlow model.
# Note: You can also register a custom `znjson` de/serializer and use `zn.outs` instead.
# 
# In this simple example we only define the epochs as parameters. For a more advanced Node you would try to catch all parameters, such as layer types, neurons, ... as `zn.params`.

# In[ ]:


class MLModel(Node):
    # dependencies
    train_data: DataPreprocessor = zn.deps(DataPreprocessor)
    # outputs
    training_history = zn.plots()
    metrics = zn.metrics()
    # custom model output
    model = TFModel()
    # parameter
    epochs = zn.params(10)

    def __init__(self, epochs: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.epochs = epochs

        self.optimizer = "adam"

    def run(self):
        """Primary Node Method"""
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
            self.train_data.features,
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


