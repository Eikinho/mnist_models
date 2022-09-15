from src.data.Data import Data

import tensorflow as tf
from tensorflow.math import confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class Net(Data):
    def __init__(self, model=None, name=None):
        super().__init__()
        self.name = name
        self.model = model

    def train(self, val_split, batch_size, epochs, data_augmentation=False):
        callback_es = EarlyStopping(
            monitor="val_loss", patience=20, restore_best_weights=True
        )
        if not data_augmentation:
            self.history = self.model.fit(
                x=self.x_train,
                y=self.y_train,
                validation_split=val_split,
                validation_data=(self.x_test, self.y_test),
                batch_size=batch_size,
                epochs=epochs,
                callbacks=[callback_es],
            )
        else:
            self.history = self.model.fit(
                x=self.train_generator,
                validation_split=val_split,
                validation_data=(self.x_test, self.y_test),
                steps_per_epoch=len(self.x_test) // batch_size,
                epochs=epochs,
                callbacks=[callback_es],
            )

        self.model.save(f"src/results/{self.name}.h5")

    def load_model(self, path):
        self.model = tf.keras.models.load_model(path)

    def evaluate(self):
        return self.model.evaluate(self.x_test, self.y_test)

    def plot_confusion_matrix(self, labels, predictions):
        confusion_mtx = confusion_matrix(labels, np.argmax(predictions, axis=1))
        _, ax = plt.subplots(figsize=(15, 10))
        ax = sns.heatmap(confusion_mtx, annot=True, fmt="d", ax=ax, cmap="Blues")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title("Confusion Matrix")
        plt.show()
