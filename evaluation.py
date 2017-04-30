import numpy as np

from vgg16 import VGGNet
from vgg5 import VGGNet5
from resnet import ResNet50 as ResNet
from utils import load_mnist


def load_models():
    """Load models """
    model_list = []

    model_list.append(VGGNet("model/vggnet.h5"))
    model_list.append(ResNet("model/resnet.h5"))
    model_list.append(VGGNet5("model/vggnet5.h5"))

    return model_list


def evaluate(prediction, true_labels):
    """Return an accuracy

    Parameters
    ----------
    prediction : 2-D array, shape (n_sample, n_classes)
        Onehot encoded predicted array

    true_labels : 2-D array, shape (n_sample, n_classes)
        Onehot encoded true array

    Returns
    ----------
    accuracy : float
        Return an accuracy
    """
    pred = np.argmax(prediction, 1)
    true = np.argmax(true_labels, 1)

    equal = (pred == true)

    return np.mean(equal)


def main():
    model_list = load_models()
    _, _, (X_test, y_test) = load_mnist()

    pred_list = []

    for idx, model in enumerate(model_list):
        pred = model.predict(X_test)
        pred_list.append(pred)

        # Check a single model accuracy
        acc = evaluate(pred, y_test)
        print("Model-{}: {:>.5%}".format(idx, acc))

    pred_list = np.asarray(pred_list)
    pred_mean = np.mean(pred_list, 0)

    accuracy = evaluate(pred_mean, y_test)
    print("Final Test Accuracy: {:>.5%}".format(accuracy))


if __name__ == '__main__':
    main()
