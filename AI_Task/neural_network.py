# neural_network.py

import importlib

# pyright: reportMissingModuleSource=false


def _load_keras_components():
    """
    Load Keras components from TensorFlow or the standalone Keras package.
    """
    try:
        keras_models = importlib.import_module("tensorflow.keras.models")
        keras_layers = importlib.import_module("tensorflow.keras.layers")
    except ImportError:
        keras_models = importlib.import_module("keras.models")
        keras_layers = importlib.import_module("keras.layers")

    return keras_models.Sequential, keras_layers.Dense


Sequential, Dense = _load_keras_components()


def build_model():
    """
    Create a simple Neural Network.
    """

    model = Sequential()

    # Input Layer + Hidden Layer
    model.add(
        Dense(
            units=8,
            activation="relu",
            input_shape=(1,)
        )
    )

    # Hidden Layer
    model.add(
        Dense(
            units=4,
            activation="relu"
        )
    )

    # Output Layer
    model.add(
        Dense(
            units=1,
            activation="sigmoid"
        )
    )

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


def train_model(model, X_train, y_train):
    """
    Train Neural Network.
    """

    history = model.fit(
        X_train,
        y_train,
        epochs=100,
        verbose=0
    )

    return model, history