# predictmodel.py

# loading json and creating model
from keras.models import model_from_json # type: ignore
from pydub import AudioSegment 
import librosa
import os
import pandas as pd
import librosa
import glob
import keras
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import optimizers # type: ignore

SAMPLE_RATE = 44100

# Function to extract MFCC features from an audio file
def get_mfcc(path):
    b, _ = librosa.core.load(path, sr=SAMPLE_RATE)
    assert _ == SAMPLE_RATE
    try:
        gmm = librosa.feature.mfcc(b, sr=SAMPLE_RATE, n_mfcc=20)
        spectral_centroids = librosa.feature.spectral_centroid(b, sr=SAMPLE_RATE)[0]
        return pd.Series(np.hstack((np.mean(gmm, axis=1), np.std(gmm, axis=1), np.mean(spectral_centroids), np.std(spectral_centroids))))
    except Exception as e:
        print('Error processing file:', path, '|', e)
        return pd.Series([0] * 42)

# Function to load the model and make predictions
def process(path):
    # Load the saved model architecture from JSON file
    json_file = open('saved_models/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    
    # Create the model from the loaded JSON
    loaded_model = model_from_json(loaded_model_json)
    
    # Load the model weights
    loaded_model.load_weights("saved_models/Voice_Pathology.h5")
    print("Loaded model from disk")
    
    # Compile the model
    opt = keras.optimizers.rmsprop(lr=0.00001, decay=1e-6)
    loaded_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    # Extract MFCC features from the audio file
    mfcc = get_mfcc(path)
    print("MFCC features:", mfcc.tolist())

    # Prepare the features for prediction
    feature_live = mfcc.tolist()
    live_df = pd.DataFrame(data=[feature_live])

    # Make predictions
    live_preds = loaded_model.predict(live_df, batch_size=32, verbose=1)
    y_pred = live_preds.argmax(axis=1)

    # Map the predicted label index to the corresponding class label
    labels = ["Healthy", "hyperkinetic_dysphonia", "hypokinetic_dysphonia", "reflux_laryngitis"]
    prediction_label = labels[int(y_pred)]

    print("Predicted label index:", y_pred)
    print("Predicted class label:", prediction_label)

    return prediction_label

# Example usage:
# print("Prediction:", process("data/Healthy/4-a_h.wav"))
