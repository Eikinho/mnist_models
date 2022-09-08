from src.models.Net import Net
import tensorflow as tf
from keras.layers import Flatten, Dense, Conv2D


class CNN_Aug(Net):
    def __init__(self):
        super().__init__(model=tf.keras.models.Sequential(), name="CNN_Aug")
        self.model.add(Flatten())
        self.model.add(Dense(32, activation=tf.nn.relu))
        self.model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
        self.model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
