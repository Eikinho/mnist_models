from src.models.ANN import ANN
from src.models.CNN import CNN
from src.models.CNN_Aug import CNN_Aug
from src.models.LeNet import LeNet


def train_models(ann_=False, cnn_=False, cnn_aug_=False, lenet_aug_=False):
    if ann_:
        ann = ANN()
        ann.train(0.2, 13, 3)
        ann.plot_history()

    if cnn_:
        cnn = CNN()
        cnn.train(0.2, 64, 10)
        cnn.plot_history()

    if cnn_aug_:
        cnn_aug = CNN_Aug()
        cnn_aug.augment_data()
        cnn_aug.train(0.2, 128, 25, True)
        cnn_aug.plot_history()

    if lenet_aug_:
        lenet_aug = LeNet()
        lenet_aug.augment_data()
        lenet_aug.train(0.2, 128, 25, True)
        lenet_aug.plot_history()


if __name__ == "__main__":
    train_models(ann_=True, cnn_=True, cnn_aug_=True, lenet_aug_=True)
