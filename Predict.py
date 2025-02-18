# Predict.py
import os
import pandas as pd
import numpy as np
import csv
from pydub import AudioSegment 
import librosa
import glob
from sklearn.linear_model import LogisticRegression

SAMPLE_RATE = 44100
#returns mfcc features with mean and standard deviation along time
def get_mfcc(path):
    b, _ = librosa.core.load(path, sr = SAMPLE_RATE)
    assert _ == SAMPLE_RATE
    try:
        gmm = librosa.feature.mfcc(y=b, sr = SAMPLE_RATE, n_mfcc=20)
        print(gmm)
        spectral_centroids = librosa.feature.spectral_centroid(y=b, sr=SAMPLE_RATE)[0]
        print("-------------")
        print(spectral_centroids)
        return pd.Series(np.hstack((np.mean(gmm, axis=1), np.std(gmm, axis=1))))
    except:
        print('bad file')
        return pd.Series([0]*40)

def process(path):
        mfcc=[]
        s=get_mfcc(path)
        print(s.tolist())
        mfcc.append(s.tolist())
        full=[]
        for i in range(0,len(mfcc)):
                print(mfcc[i])
                d=[]
                for j in mfcc[i]:
                        print(j)
                        d.append(j)
                full.append(d)
        print(full)
        X_train=pd.read_csv("dataset.csv",usecols=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39],header=None)  
        y_train=pd.read_csv("dataset.csv",usecols=[40],header=None)
        X_test=full
        model2=LogisticRegression(random_state = 0)
        model2.fit(X_train, y_train)
        y_pred = model2.predict(X_test)
        print("predicted")
        print(y_pred)
        result=""
        if y_pred[0]==1:
            result="Hyerkinetic Dysphonia \n\n\nCauses: \n1. Hyperkinetic dysphonia can result from various neurological conditions, including spasmodic dysphonia, dystonia, or Tourette syndrome. \n2. It's characterized by involuntary spasms or jerky movements of the vocal cords. \n\n\nPrecautionary Measures: \n1. Avoid triggers that exacerbate symptoms, such as stress or fatigue. \n2. Practice relaxation techniques to manage stress and tension.\n\n\nTreatment: \n1. Botox injections into the vocal cords to reduce spasms. \n2. Speech therapy focusing on techniques to control vocal cord movements. \n3. Medications to manage underlying neurological conditions."
        elif y_pred[0]==2:
            result="Hypokinetic Dysphonia \n\n\nCauses: \n1. Hypokinetic dysphonia often results from neurological conditions such as Parkinson's disease. \n2. It's characterized by reduced vocal cord movement and may lead to a breathy or hoarse voice.\n\n\nPrecautionary Measures: \n1. Manage underlying neurological conditions with medication and therapy. \n2. Work with a speech-language pathologist to learn techniques for improving voice projection and clarity. \n\n\nTreatment: \n1. Speech therapy focusing on voice exercises and respiratory support. \n2. Medications to manage symptoms of Parkinson's disease or other underlying conditions."      
        elif y_pred[0]==3:
            result="Reflux Laryngitis\n\n\nCauses: \n1. Reflux laryngitis occurs when stomach acid backs up into the throat, irritating the larynx. \n2. It's often associated with gastroesophageal reflux disease (GERD) or Laryngopharyngeal reflux (LPR). \n\n\nPrecautionary Measures: \n1. Avoid trigger foods and drinks that can worsen reflux, such as caffeine, spicy foods, alcohol, and acidic foods. \n2. Eat smaller, more frequent meals to reduce stomach pressure. \n3. Avoid lying down or bending over immediately after eating. \n\n\nTreatment: \n1. Dietary and lifestyle changes to reduce reflux symptoms. \n2. Medications to reduce stomach acid production or neutralize acid. \n3. In severe cases, surgery may be necessary to correct underlying anatomical issues or strengthen the lower esophageal sphincter."
        
        else:
                result="Healthy"
                
        return result
        
        
#process("data/Stressed/1-the_dog_is_in_front_of_the_horse.wav")

