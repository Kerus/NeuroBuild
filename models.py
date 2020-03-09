from tensorflow.keras import datasets, layers, models, optimizers, metrics, utils
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.utils import Sequence


def prepare_data(dataset):

    if dataset == 'Fashion Mnist':
        fashion_mnist = datasets.fashion_mnist
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

        X_train = X_train / 255.0
        X_test = X_test / 255.0

        class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                       "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    elif dataset == 'CIFAR-10':
        cifar = datasets.cifar10
        (X_train, y_train), (X_test, y_test) = cifar.load_data()
        X_train = X_train / 255.0
        X_test = X_test / 255.0

    y_train = utils.to_categorical(y_train, 10)
    y_test = utils.to_categorical(y_test, 10)

    return X_train, y_train, X_test, y_test



def build_simple_model(dataset='Fashion Mnist', opt='sgd', hidden=None, funcs=None, loss=None, metrics_list=None):
    model = models.Sequential()
    if dataset == 'CIFAR-10':
        model.add(layers.Flatten(input_shape=[32, 32, 3]))
    elif('Fashion Mnist'):
        model.add(layers.Flatten(input_shape=[28, 28]))
    for i in hidden.keys():
        model.add(layers.Dense(hidden[i], activation=funcs[i].lower()))
    model.add(layers.Dense(10, activation="softmax"))

    loss_dict = {
        'Categorical Crossentropy': 'categorical_crossentropy',
        'Binary Crossentropy' : 'binary_crossentropy',
        'Categorical Hinge': 'categorical_hinge',
        'Huber loss': 'huber_loss'
    }
    metrics_dict = {
        'auc': metrics.AUC(),
        'recall': metrics.Recall(),
        'accuracy': metrics.CategoricalAccuracy() if loss.startswith('Categorical') else metrics.Accuracy(),
        'precision': metrics.Precision(),
        'categorical Hinge': metrics.CategoricalHinge(),
        'squared Hinge': metrics.SquaredHinge(),
        'Kullback-Leibler divergence':metrics.KLDivergence(),
        'mean absolute error': metrics.MeanAbsoluteError(),
        'mean squared error': metrics.MeanSquaredError()
    }
    if metrics_list is not None and len(metrics_list) > 0:
        metrics_list = [metrics_dict.get(m, m) for m in metrics_list]
    else:
        metrics_list = ['accuracy']

    loss_f = loss_dict.get(loss)

    model.compile(loss=loss_f,
                  optimizer=opt,
                  metrics=metrics_list)
    return model


def load_simple_model(model_path='', weights_path='', opt='sgd', loss=None, metrics_list=None):
    model = models.load_model(model_path)
    model.load_weights(weights_path)
    loss_dict = {
        'Categorical Crossentropy': 'categorical_crossentropy',
        'Binary Crossentropy': 'binary_crossentropy',
        'Categorical Hinge': 'categorical_hinge',
        'Huber loss': 'huber_loss'
    }
    metrics_dict = {
        'auc': metrics.AUC(),
        'recall': metrics.Recall(),
        'accuracy': metrics.CategoricalAccuracy() if loss.startswith('Categorical') else metrics.Accuracy(),
        'precision': metrics.Precision(),
        'categorical Hinge': metrics.CategoricalHinge(),
        'squared Hinge': metrics.SquaredHinge(),
        'Kullback-Leibler divergence': metrics.KLDivergence(),
        'mean absolute error': metrics.MeanAbsoluteError(),
        'mean squared error': metrics.MeanSquaredError()
    }
    if metrics_list is not None and len(metrics_list) > 0:
        metrics_list = [metrics_dict.get(m, m) for m in metrics_list]
    else:
        metrics_list = ['accuracy']

    loss_f = loss_dict.get(loss)

    model.compile(loss=loss_f,
                  optimizer=opt,
                  metrics=metrics_list)
    return model


def print_model_results(model, X, y, train=True):
    if train:
        st.write('Results on train data')
    else:
        st.write('Results on test data')
    results = model.evaluate(X, y)
    results = [np.round(float(res), 2) for res in results]
    metrics = dict(zip(model.metrics_names, results))
    st.write(metrics)


def plot_model_history(history):
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)  # set the vertical range to [0-1]
    plt.show()
    st.pyplot()

