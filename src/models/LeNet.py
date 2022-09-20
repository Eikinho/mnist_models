from src.models.Net import Net

import tensorflow as tf
from keras.layers import Flatten, Dense, Conv2D, AveragePooling2D


class LeNet(Net):
    def __init__(self):
        super().__init__(model=tf.keras.models.Sequential(), name="LeNet_Aug")
        self.model.add(
            Conv2D(16, (5, 5), activation="sigmoid", input_shape=(28, 28, 1))
        )
        self.model.add(
            Conv2D(16, (5, 5), activation="sigmoid", input_shape=(26, 26, 1))
        )
        self.model.add(AveragePooling2D(13, 13))
        self.model.add(Conv2D(16, (11, 11), activation=tf.nn.sigmoid))
        self.model.add(AveragePooling2D(5, 5))
        self.model.add(Flatten())
        self.model.add(Dense(120, activation=tf.nn.sigmoid))
        self.model.add(Dense(84, activation=tf.nn.sigmoid))
        self.model.add(Dense(10, activation=tf.nn.softmax))
        self.model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
