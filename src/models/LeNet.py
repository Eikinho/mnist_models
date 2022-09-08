from src.models.Net import Net
import tensorflow as tf
from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D


class LeNet(Net):
    def __init__(self):
        super().__init__(model=tf.keras.models.Sequential(), name="LeNet")
        self.model.add(
            Conv2D(16, (5, 5), activation="sigmoid", input_shape=(28, 28, 1))
        )
        self.model.add(MaxPooling2D(2, 2))
        self.model.add(Conv2D(16, (5, 5), activation="sigmoid"))
        self.model.add(MaxPooling2D(2, 2))
        self.model.add(Flatten())
        self.model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
        self.model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
