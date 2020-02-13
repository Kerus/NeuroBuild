from tensorflow.keras import datasets, layers, models, optimizers
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def build_simple_model(opt='sgd'):
    model = models.Sequential()
    model.add(layers.Flatten(input_shape=[28, 28]))
    model.add(layers.Dense(300, activation="relu"))
    model.add(layers.Dense(100, activation="relu"))
    model.add(layers.Dense(10, activation="softmax"))

    model.compile(loss="sparse_categorical_crossentropy",
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
