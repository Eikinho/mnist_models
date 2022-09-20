from src.models.Net import Net

import tensorflow as tf
from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D


class CNN_Aug(Net):
    def __init__(self):
        super().__init__(model=tf.keras.models.Sequential(), name="CNN_Aug")
        self.model.add(Conv2D(6, (3, 3), activation="sigmoid", input_shape=(28, 28, 1)))
        self.model.add(MaxPooling2D(2, 2))
        self.model.add(Conv2D(16, (3, 3), activation="sigmoid"))
        self.model.add(MaxPooling2D(2, 2))
        self.model.add(Flatten())
        self.model.add(Dense(120, activation=tf.nn.sigmoid))
        self.model.add(Dense(84, activation=tf.nn.sigmoid))
        self.model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
        self.model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
