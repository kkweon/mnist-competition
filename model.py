import os

from keras.layers import Input
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from utils import train_generator


class BaseModel(object):
    """Base Model Interface

    Methods
    ----------
    fit(train_data, valid_data, epohcs, batch_size, **kwargs)
    predict(X)
    evaluate(X, y)

    Examples
    ----------
    >>> model = Model("example", inference, "model.h5")
    >>> model.fit([X_train, y_train], [X_val, y_val])
    """

    def __init__(self, name, fn, model_path):
        """Constructor for BaseModel

        Parameters
        ----------
        name : str
            Name of this model

        fn : function
            Inference function, y = fn(X)

        model_path : str
            Path to a model.h5
        """
        X = Input(shape=[28, 28, 1])
        y = fn(X)

        self.model = Model(X, y, name=name)
        self.model.compile("adam", "categorical_crossentropy", ["accuracy"])

        self.path = model_path
        self.name = name
        self.load()

    def fit(self, train_data, valid_data, epochs=10, batchsize=32, **kwargs):
        """Training function

        Evaluate at each epoch against validation data
        Save the best model according to the validation loss

        Parameters
        ----------
        train_data : tuple, (X_train, y_train)
            X_train.shape == (N, H, W, C)
            y_train.shape == (N, N_classes)

        valid_data : tuple
            (X_val, y_val)

        epochs : int
            Number of epochs to train

        batchsize : int
            Minibatch size

        **kwargs
            Keywords arguments for `fit_generator`
        """
        callback_best_only = ModelCheckpoint(self.path, save_best_only=True)
        train_gen, val_gen = train_generator()

        X_train, y_train = train_data
        X_val, y_val = valid_data

        N = X_train.shape[0]
        N_val = X_val.shape[0]

        self.model.fit_generator(train_gen.flow(X_train, y_train, batchsize),
                                 steps_per_epoch=N / batchsize,
                                 validation_data=val_gen.flow(X_val, y_val, batchsize),
                                 validation_steps=N_val / batchsize,
                                 epochs=epochs,
                                 callbacks=[callback_best_only],
                                 **kwargs)

    def save(self):
        """Save weights

        Should not be used manually
        """
        self.model.save_weights(self.path)

    def load(self):
        """Load weights from self.path """
        if os.path.isfile(self.path):
            self.model.load_weights(self.path)
            print("Model loaded")
        else:
            print("No model is found")

    def predict(self, X):
        """Return probabilities for each classes

        Parameters
        ----------
        X : array-like (N, H, W, C)

        Returns
        ----------
        y : array-like (N, N_classes)
            Probability array
        """
        return self.model.predict(X)

    def evaluate(self, X, y):
        """Return an accuracy

        Parameters
        ----------
        X : array-like (N, H, W, C)
        y : array-like (N, N_classes)

        Returns
        ----------
        acc : float
            Accuracy
        """
        return self.model.evaluate(X, y)
