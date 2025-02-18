# trainmodel.py

import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.pyplot import specgram
import keras
import itertools
from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten, Conv1D, LSTM, Bidirectional
from keras.layers import GlobalMaxPooling1D, MaxPooling1D
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
from keras import optimizers
import os
import pandas as pd
import glob 
import scipy.io.wavfile
import sys
from sklearn.utils import shuffle
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
import json
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from keras.models import model_from_json
import csv
from numpy import zeros
import keras
import keras.utils
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout
from keras.layers import TimeDistributed, LSTM



def process(path):

    X = pd.read_csv(path,usecols=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39],header=None)
    y = pd.read_csv(path,usecols=[40],header=None)
    y = to_categorical(y)
    print("to_categorical(y)==",to_categorical(y))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)


    #print(X_train)
    #print(y_train)
    x_traincnn =np.expand_dims(X_train, axis=2)
    x_testcnn= np.expand_dims(X_test, axis=2)

    model = Sequential()
    model.add(Conv1D(256, 5,padding='same',input_shape=(40,1)))
    model.add(Activation('relu'))
    model.add(Conv1D(128, 5,padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(MaxPooling1D(pool_size=(8)))
    model.add(Conv1D(128, 5,padding='same',))
    model.add(Activation('relu'))
    model.add(Conv1D(128, 5,padding='same',))
    model.add(Activation('relu'))
    model.add(Flatten())
    #model.add(Flatten())
    model.add(Dense(4))
    model.add(Activation('softmax'))
    opt = keras.optimizers.RMSprop(lr=0.00001, decay=1e-6)
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer=opt,metrics=['accuracy'])
    cnnhistory=model.fit(x_traincnn, y_train, batch_size=16, epochs=16, validation_data=(x_testcnn, y_test))

    fig = plt.figure(0)
    plt.plot(cnnhistory.history['loss'])
    plt.plot(cnnhistory.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    fig.savefig('results/Accuracy.png')
    plt.pause(5)
    plt.show(block=False)
    plt.close()	

    model_name = 'Voice_Pathology.h5'
    save_dir = os.path.join(os.getcwd(), 'saved_models')
    # Save model and weights
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)

    model_json = model.to_json()
    with open("saved_models/model.json", "w") as json_file:
        json_file.write(model_json)

    json_file = open('saved_models/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    #process("dataset.csv")
