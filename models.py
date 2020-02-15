from tensorflow.keras import datasets, layers, models, optimizers
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.utils import Sequence


def build_simple_model(dataset='fashion_mnist', opt='sgd', hidden=None, funcs=None, loss=None):
    model = models.Sequential()
    if dataset == 'CIFAR-10':
        model.add(layers.Flatten(input_shape=[32, 32, 3]))
    elif('Fashion Mnist'):
        model.add(layers.Flatten(input_shape=[28, 28]))
    for i in hidden.keys():
        model.add(layers.Dense(hidden[i], activation=funcs[i].lower()))
    model.add(layers.Dense(10, activation="softmax"))

    loss_dict = {
        'Categorical Crossentropy': 'sparse_categorical_crossentropy',
        'Binary Crossentropy' : 'binary_crossentropy',
        'Categorical Hinge': 'categorical_hinge',
        'Huber loss': 'huber_loss'
    }
    loss_f = loss_dict.get(loss)

    model.compile(loss=loss_f,
                  optimizer=opt,
                  metrics=['accuracy'])
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
