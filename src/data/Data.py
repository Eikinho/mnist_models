from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt


class Data:
    def __init__(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()

    def augment_data(self, BATCH_SIZE=128):
        train_datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            rotation_range=40,
            width_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=False,
        )

        validation_datagen = ImageDataGenerator(rescale=1.0 / 255)

        train_generator = train_datagen.flow(
            self.x_train, self.y_train, batch_size=BATCH_SIZE
        )

        validation_generator = validation_datagen.flow(
            self.x_test, self.y_test, batch_size=BATCH_SIZE
        )

        (self.x_train, self.y_train), (self.x_test, self.y_test) = (
            train_generator,
            validation_generator,
        )

    def plot_data(self, x_train, y_train):
        plt.figure(figsize=(10, 10))
        for i in range(25):
            plt.subplot(5, 5, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(x_train[i], cmap=plt.cm.binary)
            plt.xlabel(y_train[i])
        plt.show()
