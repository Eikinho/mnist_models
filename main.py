from src.data.Data import Data
from src.models.Net import Net
from src.models.ANN import ANN
from src.models.CNN import CNN

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def main():
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


if __name__ == "__main__":
    ANN = ANN()
    ANN.train(0.2, 8, 5)

    CNN = CNN()
    CNN.train(0.2, 8, 5)
