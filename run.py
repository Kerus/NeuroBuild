import streamlit as st
from tensorflow.keras import datasets, layers, models, optimizers, utils
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from models import *

# preprocessing
from sklearn.model_selection import train_test_split

#data_path = st.sidebar.file_uploader('file', type='csv')
st.title('Neuro Build')

task_option = st.sidebar.radio('Choose a task', ('Dataset overview', 'Feed-forward networks',
                                                           'Convolutional networks', 'AutoEncoders',
                                                           'GAN', 'Recurrent networks'))

dataset = st.selectbox('choose dataset', ('Fashion Mnist', 'CIFAR-10'))

#if data_path is not None:
#    df = st.cache(pd.read_csv)(data_path)
if task_option == 'Dataset overview':

    st.header('Dataset overview')
    st.write('Wait for it...')


elif task_option == 'Feed-forward networks':

    st.header('Construct feed-forward network')
    st.write('input and output layers are currently set by default')
    hidden = st.number_input('Count of hidden layers: ', value=2, max_value=10, min_value=1)
    i_layers = dict()
    i_act_funcs = dict()
    st.write(
        f'<div style="height:1px;border:solid 1px #cccccc;margin-bottom:10px;"></div>',
        unsafe_allow_html=True
    )
    for i in range(hidden):
        i_layers[i] = st.number_input('Hidden layer {:.0f}. Count of neurons: '.format(i+1),
                                      value=300, max_value=10000, min_value=1)
        i_act_funcs[i] = st.selectbox('Activation function of layer {:.0f}'.format(i+1),
                                      ('ReLU', 'SELU', 'Tanh', 'Sigmoid', 'Linear'))

        st.write(
            f'<div style="height:1px;border:solid 1px #cccccc;margin-bottom:40px;"></div>',
            unsafe_allow_html=True
        )
    st.write('Adjust other hyper-parameters')
    opt = st.selectbox('Optimiser', ('sgd', 'adadelta', 'adagrad', 'adam', 'adamax'))
    epochs = st.number_input('Epochs: ', value=30, max_value=10000, min_value=1)
    val_perc = st.number_input('Percent of validation data: ', value=10, max_value=70, min_value=5)
    val_split = val_perc / 100
    loss_f = st.selectbox('Loss function', ('Categorical Crossentropy', 'Binary Crossentropy',
                                            'Categorical Hinge', 'Huber loss'))

    if st.button("Train Network"):

        with st.spinner('network is training...'):
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

            if loss_f != 'Categorical Crossentropy':
                y_train = utils.to_categorical(y_train, 10)
                y_test = utils.to_categorical(y_test, 10)

            model = build_simple_model(dataset=dataset, opt=opt, hidden=i_layers, funcs=i_act_funcs, loss=loss_f)

            history = model.fit(X_train, y_train, epochs=epochs, validation_split=val_split, shuffle=True)

        st.success('Training is finished')

        plot_model_history(history)
        print_model_results(model, X_train, y_train, train=True)
        print_model_results(model, X_test, y_test, train=False)

elif task_option == 'Convolutional networks':

    st.header('Convolutional networks')
    st.write('Wait for it...')

elif task_option == 'AutoEncoders':

    st.header('AutoEncoders')
    st.write('Wait for it...')

elif task_option == 'GAN':

    st.header('Generative adversarial networks')
    st.write('Wait for it...')

elif task_option == 'Recurrent networks':

    st.header('Recurrent networks')
    st.write('Wait for it...')
