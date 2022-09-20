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
            rotation_range=10,
            zoom_range=0.1,
            width_shift_range=0.25,
            height_shift_range=0.25,
            horizontal_flip=False,
            vertical_flip=False,
            fill_mode='constant',
            cval=0.0
        )

        self.train_generator = train_datagen.flow(
            self.x_train, self.y_train, batch_size=BATCH_SIZE
        )


    def plot_some_data(self, imgs):
        fig, axs = plt.subplots(1, 5, figsize=(15, 15))
        axs = axs.flatten()
        for img, ax in zip(imgs, axs):
            ax.imshow(img.reshape(28, 28), cmap="gray")
            ax.axis("off")
        plt.tight_layout()
        plt.show()
