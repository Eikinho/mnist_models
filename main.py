from src.models.ANN import ANN
from src.models.CNN import CNN
from src.models.CNN_Aug import CNN_Aug
from src.models.LeNet import LeNet

import cv2 as cv
import numpy as np
from pathlib import Path


def test_models():
    raiz_desafio = Path(
        "/content/drive/MyDrive/Insper/Visao/Visão-2022-Compartilhada com alunos/atividade_batalha_das_redes/digitos_para_teste"
    )
    dict_desafios = {}
    for arq in raiz_desafio.iterdir():
        digito = str(arq).split(".")[0][-1]
        dict_desafios[digito] = {}
        dict_desafios[digito]["true"] = int(digito)
        dict_desafios[digito]["file"] = arq
        im = cv.imread(str(arq.resolve()))
        b, g, r = cv.split(im)
        p = np.array(b).astype("float32")
        dict_desafios[digito]["imagem"] = p / 255
        dict_desafios[digito]["predito"] = 0


def train_models():
    # ann = ANN()
    # ann.train(0.2, 8, 5)
    # ANN.evaluate()
    # ANN.plot_confusion_matrix(ANN.y_test, ANN.predictions)

    # cnn = CNN()
    # cnn.train(0.2, 128, 25)
    # CNN.evaluate()
    # CNN.plot_confusion_matrix(CNN.y_test, CNN.predictions)

    cnn_aug = CNN_Aug()
    cnn_aug.augment_data()
    cnn_aug.train(0.2, 128, 25, True)
    # CNN_Aug.evaluate()
    # CNN_Aug.plot_confusion_matrix(CNN_Aug.y_test, CNN_Aug.predictions)

    lenet_aug = LeNet()
    lenet_aug.augment_data()
    # lenet_aug.train(0.2, 128, 25, True)
    # LeNet_Aug.evaluate()
    # LeNet_Aug.plot_confusion_matrix(LeNet_Aug.y_test, LeNet_Aug.predictions)


if __name__ == "__main__":
    train_models()
    # test_models()
