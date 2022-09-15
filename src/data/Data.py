from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np


class Data:
    def __init__(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()
        self.x_train = np.reshape(self.x_train, [-1, 28, 28, 1])
        self.x_test = np.reshape(self.x_test, [-1, 28, 28, 1])
        print(f"size before augmentation: {self.x_train.shape}")

    def augment_data(self, BATCH_SIZE=128):
        train_datagen = ImageDataGenerator(
            featurewise_center=True,
            featurewise_std_normalization=True,
            rotation_range=10,
            fill_mode="nearest",
            validation_split=0.15,
        )

        validation_datagen = ImageDataGenerator(rescale=1.0 / 255)

        # Flow training images in batches of 128 using train_datagen generator
        self.train_generator = train_datagen.flow(
            self.x_train, self.y_train, batch_size=BATCH_SIZE
        )

        # Flow validation images in batches of 128 using test_datagen generator
        validation_generator = validation_datagen.flow(
            self.x_test, self.y_test, batch_size=BATCH_SIZE
        )

    def plot_data(self, x_train, y_train):
        plt.figure(figsize=(10, 10))
        for i in range(25):
            plt.subplot(5, 5, i + 1)
            plt.grid(False)
            plt.imshow(x_train[i], cmap=plt.cm.binary)
            plt.xlabel(y_train[i])
        plt.show()
