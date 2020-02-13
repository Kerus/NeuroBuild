import streamlit as st
from tensorflow.keras import datasets, layers, models, optimizers
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
#if data_path is not None:
#    df = st.cache(pd.read_csv)(data_path)
if task_option == 'Dataset overview':

    st.header('Dataset overview')
    st.write('Wait for it...')


elif task_option == 'Feed-forward networks':

    st.header('Feed-forward network')

    dataset = st.selectbox('choose dataset', ('Fashion Mnist', 'CIFAR'))

    opt = st.selectbox('Optimiser', ('sgd', 'adadelta', 'adagrad', 'adam', 'adamax'))
    epochs = st.number_input('Epochs: ', value=30, max_value=10000, min_value=1)
    val_perc = st.number_input('Percent of validation data: ', value=10, max_value=70, min_value=5)
    val_split = val_perc / 10

    if st.button("Train Network"):

        with st.spinner('network is training...'):
            if dataset == 'Fashion Mnist':
                fashion_mnist = datasets.fashion_mnist
                (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

                X_train = X_train / 255.0
                X_test = X_test / 255.0

                class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

            model = build_simple_model(opt=opt)
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
