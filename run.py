import streamlit as st
from tensorflow.keras import datasets, layers, models, optimizers, utils, callbacks
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import os
from models import *

# preprocessing


def file_selector(label='Select a file', folder_path='.'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox(label, filenames)
    return os.path.join(folder_path, selected_filename)

from sklearn.model_selection import train_test_split

#data_path = st.sidebar.file_uploader('file', type='csv')
st.title('Neuro Build')

task_option = st.sidebar.radio('Choose a task', ('Feed-forward networks','Convolutional networks', 'AutoEncoders',
                                                           'GAN', 'Recurrent networks'))

dataset = st.selectbox('choose dataset', ('Fashion Mnist', 'CIFAR-10'))

#if data_path is not None:
#    df = st.cache(pd.read_csv)(data_path)

if task_option == 'Feed-forward networks':

    build_option = st.radio('Build or load model', ('Construct new model', 'Load model'))

    if build_option == 'Construct new model':
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

        loss_f = st.selectbox('Loss function', ('Categorical Crossentropy', 'Binary Crossentropy',
                                                'Categorical Hinge', 'Huber loss'))
        metrics_list = st.multiselect('Metrics: ', ('accuracy', 'recall', 'precision', 'auc', 'categorical Hinge',
                                                    'squared Hinge', 'Kullback-Leibler divergence',
                                                    'mean absolute error', 'mean squared error'), ('accuracy'))

        epochs = st.number_input('Count of Epochs: ', value=30, max_value=10000, min_value=1)
        val_perc = st.number_input('Percent of validation data: ', value=10, max_value=70, min_value=5)
        val_split = val_perc / 100

        if st.button("Train Network"):

            with st.spinner('network is training...'):
                X_train, y_train, X_test, y_test = prepare_data(dataset)

                model = build_simple_model(dataset=dataset, opt=opt, hidden=i_layers,
                                           funcs=i_act_funcs, loss=loss_f, metrics_list=metrics_list)

                checkpoint_cb = callbacks.ModelCheckpoint("models_weights/my_keras_model.h5", save_best_only=True)
                early_stopping_cb = callbacks.EarlyStopping(patience=10, restore_best_weights=True)
                history = model.fit(X_train, y_train, epochs=epochs, validation_split=val_split,
                                    shuffle=True, callbacks=[checkpoint_cb, early_stopping_cb])

            st.success('Training is finished')
            model.save("models_weights/final_model.h5")
            model.save_weights("models_weights/final_model_weights.h5")
            st.success('Model and weights are stored')

            plot_model_history(history)
            st.write('Loss function: ', loss_f)
            print_model_results(model, X_train, y_train, train=True)
            print_model_results(model, X_test, y_test, train=False)

    elif build_option == 'Load model':
        data_model_path = file_selector('choose a model', 'models_weights/')
        data_weights_path = file_selector('choose a weights', 'models_weights/')
        #data_model_path = st.file_uploader('choose a model', type='h5')
        #data_weights_path = st.file_uploader('choose a model weights', type='h5')

        opt = st.selectbox('Optimiser', ('sgd', 'adadelta', 'adagrad', 'adam', 'adamax'))

        loss_f = st.selectbox('Loss function', ('Categorical Crossentropy', 'Binary Crossentropy',
                                                'Categorical Hinge', 'Huber loss'))
        metrics_list = st.multiselect('Metrics: ',
                                      ('accuracy', 'recall', 'precision', 'auc', 'categorical Hinge',
                                       'squared Hinge', 'Kullback-Leibler divergence',
                                       'mean absolute error', 'mean squared error'), ('accuracy'))
        if st.button('evaluate a model'):

            model = load_simple_model(data_model_path, data_weights_path, opt, loss_f, metrics_list)

            X_train, y_train, X_test, y_test = prepare_data(dataset)

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
