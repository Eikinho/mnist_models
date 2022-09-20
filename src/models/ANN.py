from src.models.Net import Net

import tensorflow as tf
from keras.layers import Flatten, Dense, Dropout


class ANN(Net):
    def __init__(self):
        super().__init__(model=tf.keras.models.Sequential(), name="ANN")
        self.model.add(Flatten())
        self.model.add(Dense(256, activation=tf.nn.relu))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(128, activation=tf.nn.relu))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(10, activation=tf.nn.softmax))
        self.model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
