# first script:

from pathlib import Path
import matplotlib.pyplot as plt
import time

####################################################################################

# read and import audio files
import datasetMenagment as ds

# feature extraction
import featureExtraction as fe

# classification

####################################################################################
import util

audiofiles_directory = Path("D:\Programmi\PyCharm Projects\SpeechRecognition\Comandi Samu\(1000,10)_ComandiDatasetSelezionati")
dataset_directory = Path("D:\Programmi\PyCharm Projects\SpeechRecognition")
features_directory = Path("D:\Programmi\PyCharm Projects\SpeechRecognition")

# Fixed sample rate of the audio signals analyzed
sample_rate = 16000  # Hz Sampling frequency of this particular dataset

dataset_name = 'myData.p'

# Load data:
print("Loading the dataset...")
dataset = ds.loadDataset(dataset_directory, audiofiles_directory, dataset_name=dataset_name)
print("Dataset loaded!")

mfcc_features, mfcc_dataset_noise_level = fe.loadFeatures(dataset, sample_rate, features_directory, dataset_name, method=1)
# mod 1 builds the feature vectors computing the MFCC of every signal (variable size -> has to be fixed)
rtd_features, rtd_dataset_noise_level = fe.loadFeatures(dataset, sample_rate, features_directory, dataset_name, method=2)
# mod 2 builds the feature vectors computing the RTD of every signal (fixed size)

# print("Loading the train/test sets...")
# train_dataset, test_dataset = fe.loadSubsets(features, features_directory)
# print("Train/test sets loaded!")


####################################################################################

# Data Visualization:
# util.plotClass(6, mfcc_features)
util.plotClass(6, rtd_features)

# print("dataset:")
# util.getParameters(dataset)
#
# print("mfcc_features:")
# util.getParameters(mfcc_features)
#
# print("rtd_features:")
# util.getParameters(rtd_features)
#
# util.viewExample("off0.wav", dataset, mfcc_features, rtd_features)

plt.show()
