from src.data.Data import Data

import tensorflow as tf
from sklearn.metrics import confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
import numpy as np
import pandas as pd
import cv2


class Net(Data):
    def __init__(self, name=None, model=None):
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

    def load_test_data(self, data_path="src/data/digitos_para_teste"):
        raiz_desafio = Path(
            data_path
        )
        self.dict_desafios = {}
        self.y_true = []
        self.test_imgs = []
        for arq in raiz_desafio.iterdir():
            digito = str(arq).split(".")[0][-1]
            self.y_true.append(int(digito))
            im = cv2.imread(str(arq.resolve()))
            b, _, _ = cv2.split(im)
            p = np.array(b).astype("float32")
            p = p.reshape(-1, 28, 28, 1)
            p = p / 255
            self.test_imgs.append(p)
        
    def test(self):
        self.y_pred = []
        for img in self.test_imgs:
            self.y_pred.append(np.argmax(self.model.predict(img)))
        
    def load_model(self):
        self.model = tf.keras.models.load_model(f"src/results/{self.name}.h5")

    def plot_confusion_matrix(self, labels, predictions):
        confusion_mtx = confusion_matrix(
            labels, predictions, labels=np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), normalize="true"
        )
        df_cm = pd.DataFrame(confusion_mtx, index = range(0,10), columns = range(0,10))
        plt.figure(figsize = (10,7))
        sns.heatmap(df_cm, annot=True)
        plt.show()

    def plot_history(self):
        fig = plt.figure()
        plt.subplot(2,1,1)
        plt.plot(self.history.history['accuracy'])
        plt.plot(self.history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='lower right')
        plt.subplot(2,1,2)
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper right')
        plt.tight_layout()
        plt.show()