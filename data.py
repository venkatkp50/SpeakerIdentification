import os
from config import DATA_PATH , MODEL_PATH
import pandas as pd
import glob
import pickle
import scipy.io.wavfile as wav
from python_speech_features import mfcc

def getSpeakerData():
    return pd.Series(os.listdir(DATA_PATH))

def getFileList(speaker):
    filelist = glob.glob(os.path.join(DATA_PATH,speaker)+'\*.wav')    
    return filelist

def extract_mfcc(audio_file, num_features=13):
  (sr, wave_file) = wav.read(audio_file)
  mfcc_features = mfcc(wave_file, samplerate=sr, numcep=num_features)
  return mfcc_features

def load_gmm_model(model_file):
    with open(MODEL_PATH + '\\' +model_file, 'rb') as file:
        gmm_model = pickle.load(file)
    return gmm_model

def speaker_recognition(audio_file, gmm_model):
    mfcc_features = extract_mfcc(audio_file)
    likelihood = gmm_model.score(mfcc_features)
    return likelihood

def getSpeakerLikelihood(testFile):
    speaker_list = []
    likelihood_list = []

    for speaker_name in os.listdir(DATA_PATH):
        print(speaker_name)
        model_file_to_load = f"{speaker_name}_model.pkl"
        loaded_model = load_gmm_model(model_file_to_load)
        likelihood = speaker_recognition(testFile, loaded_model)
        speaker_list.append(speaker_name)
        likelihood_list.append(likelihood)
  
        df = pd.DataFrame({'Speaker':speaker_list,'likelihood':likelihood_list})
    return df
    