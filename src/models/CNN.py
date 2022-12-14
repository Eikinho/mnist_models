from src.models.Net import Net

import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Dropout


class CNN(Net):
    def __init__(self):
        super().__init__(model=tf.keras.models.Sequential(), name="CNN")
        self.model.add(Conv2D(64, (7, 7), activation=tf.nn.relu, input_shape=(28, 28, 1)))
        self.model.add(MaxPooling2D(2, 2))
        self.model.add(Conv2D(128, (7, 7), activation=tf.nn.relu))
        self.model.add(Dropout(0.2))
        self.model.add(MaxPooling2D(2, 2))
        self.model.add(Flatten())
        self.model.add(Dense(64, activation=tf.nn.relu))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(10, activation=tf.nn.softmax))
        self.model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
