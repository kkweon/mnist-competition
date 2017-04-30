import argparse

from keras.layers import (
    Conv2D, BatchNormalization,
    MaxPooling2D, ZeroPadding2D, AveragePooling2D,
    add, Dense, Flatten
)
from keras.layers.advanced_activations import PReLU
from model import BaseModel
from utils import load_mnist


def resnet(input_tensor):
    """Inference function for ResNet

    y = resnet(X)

    Parameters
    ----------
    input_tensor : keras.layers.Input

    Returns
    ----------
    y : softmax output
    """
    def name_builder(type, stage, block, name):
        return "{}{}{}_branch{}".format(type, stage, block, name)

    def identity_block(input_tensor, kernel_size, filters, stage, block):
        F1, F2, F3 = filters

        def name_fn(type, name):
            return name_builder(type, stage, block, name)

        x = Conv2D(F1, (1, 1), name=name_fn('res', '2a'))(input_tensor)
        x = BatchNormalization(name=name_fn('bn', '2a'))(x)
        x = PReLU()(x)

        x = Conv2D(F2, kernel_size, padding='same', name=name_fn('res', '2b'))(x)
        x = BatchNormalization(name=name_fn('bn', '2b'))(x)
        x = PReLU()(x)

        x = Conv2D(F3, (1, 1), name=name_fn('res', '2c'))(x)
        x = BatchNormalization(name=name_fn('bn', '2c'))(x)
        x = PReLU()(x)

        x = add([x, input_tensor])
        x = PReLU()(x)

        return x

    def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
        def name_fn(type, name):
            return name_builder(type, stage, block, name)

        F1, F2, F3 = filters

        x = Conv2D(F1, (1, 1), strides=strides, name=name_fn("res", "2a"))(input_tensor)
        x = BatchNormalization(name=name_fn("bn", "2a"))(x)
        x = PReLU()(x)

        x = Conv2D(F2, kernel_size, padding='same', name=name_fn("res", "2b"))(x)
        x = BatchNormalization(name=name_fn("bn", "2b"))(x)
        x = PReLU()(x)

        x = Conv2D(F3, (1, 1), name=name_fn("res", "2c"))(x)
        x = BatchNormalization(name=name_fn("bn", "2c"))(x)

        sc = Conv2D(F3, (1, 1), strides=strides, name=name_fn("res", "1"))(input_tensor)
        sc = BatchNormalization(name=name_fn("bn", "1"))(sc)

        x = add([x, sc])
        x = PReLU()(x)

        return x

    net = ZeroPadding2D((3, 3))(input_tensor)
    net = Conv2D(64, (7, 7), strides=(2, 2), name="conv1")(net)
    net = BatchNormalization(name="bn_conv1")(net)
    net = PReLU()(net)
    net = MaxPooling2D((3, 3), strides=(2, 2))(net)

    net = conv_block(net, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    net = identity_block(net, 3, [64, 64, 256], stage=2, block='b')
    net = identity_block(net, 3, [64, 64, 256], stage=2, block='c')

    net = conv_block(net, 3, [128, 128, 512], stage=3, block='a')
    net = identity_block(net, 3, [128, 128, 512], stage=3, block='b')
    net = identity_block(net, 3, [128, 128, 512], stage=3, block='c')
    net = identity_block(net, 3, [128, 128, 512], stage=3, block='d')

    net = conv_block(net, 3, [256, 256, 1024], stage=4, block='a')
    net = identity_block(net, 3, [256, 256, 1024], stage=4, block='b')
    net = identity_block(net, 3, [256, 256, 1024], stage=4, block='c')
    net = identity_block(net, 3, [256, 256, 1024], stage=4, block='d')
    net = identity_block(net, 3, [256, 256, 1024], stage=4, block='e')
    net = identity_block(net, 3, [256, 256, 1024], stage=4, block='f')
    net = AveragePooling2D((2, 2))(net)

    net = Flatten()(net)
    net = Dense(10, activation="softmax", name="softmax")(net)

    return net


class ResNet50(BaseModel):
    def __init__(self, model_path):
        super(ResNet50, self).__init__("resnet", resnet, model_path)


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("epoch", type=int, help="Epochs")
    parser.add_argument("--model_path", default="model/resnet.h5", type=str, help="model path (default: model/resnet.h5)")

    args = parser.parse_args()
    return args.epoch, args.model_path


def main():
    EPOCH, MODEL_PATH = arg_parser()
    train, valid, _ = load_mnist(samplewise_normalize=True)

    model = ResNet50(MODEL_PATH)
    model.fit((train[0], train[1]), (valid[0], valid[1]), EPOCH)


if __name__ == '__main__':
    main()
