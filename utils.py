import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.examples.tutorials.mnist import input_data
from keras.utils import plot_model
from keras.models import load_model


def normalize_image_array(image_array):
    """Samplewise Normalize images

    Each image is subtracted by its mean value
    and divided by its standard deviation

    Parameters
    ----------
    image_array : 2-d array, shape (N_sample, D_input)
        Standardize input

    Returns
    ----------
    normalized image : 2-d array, shape (N_sample, D_input)
    """
    N, D = image_array.shape

    numerator = image_array - np.expand_dims(np.mean(image_array, 1), 1)
    denominator = np.expand_dims(np.std(image_array, 1), 1)

    return numerator / (denominator + 1e-7)


def load_mnist(samplewise_normalize=True):
    """Load mnist data and reshape

    Parameters
    ----------
    samplewise_normalize : bool (optional)
        Normalize images

    Returns
    ----------
    (train_X, train_y) : (4-d array, 2-d array)
    (valid_X, valid_y) : (4-d array, 2-d array)
    (test_X, test_y) : (4-d array, 2-d array)
    """
    mnist = input_data.read_data_sets("MNIST/", one_hot=True)

    train_X = mnist.train.images
    train_y = mnist.train.labels

    valid_X = mnist.validation.images
    valid_y = mnist.validation.labels

    test_X = mnist.test.images
    test_y = mnist.test.labels

    if samplewise_normalize:
        train_X = normalize_image_array(train_X)
        valid_X = normalize_image_array(valid_X)
        test_X = normalize_image_array(test_X)

    train_X = np.reshape(train_X, [-1, 28, 28, 1])
    valid_X = np.reshape(valid_X, [-1, 28, 28, 1])
    test_X = np.reshape(test_X, [-1, 28, 28, 1])

    return (train_X, train_y), (valid_X, valid_y), (test_X, test_y)


def train_generator():
    """Train Generator for Keras

    Returns
    ----------
    train_gen : generator
        Yield augmented images

    val_gen : generator
        Yield non-augmented images
    """
    train_gen = ImageDataGenerator(
        rotation_range=30,
        shear_range=0.1,
        zoom_range=0.1,
        width_shift_range=0.2,
        height_shift_range=0.2,
    )

    val_gen = ImageDataGenerator()
    return train_gen, val_gen


def plot_(model_path, file_path):
    """Visualize a model

    Parameters
    ----------
    model_path : str
        Path to the model.h5

    file_path : str
        Destination file to save
        i.e. model.png
    """
    model = load_model(model_path)
    plot_model(model,
               file_path,
               show_shapes=True,
               show_layer_names=False)


if __name__ == '__main__':
    model_list = ["model/vggnet.h5", "model/resnet.h5", "model/vggnet5.h5"]
    file_list = ["images/vggnet.png", "images/resnet.png", "images/vggnet5.png"]

    [plot_(model_path, file_path) for model_path, file_path in zip(model_list, file_list)]
