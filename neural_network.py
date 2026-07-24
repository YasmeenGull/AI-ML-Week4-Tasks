# neural_network.py

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


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