# first script:

from pathlib import Path
import matplotlib.pyplot as plt
import time

####################################################################################

# read and import audio files
import datasetMenagment as ds

# pre-processing
import preProcessing as sp

# feature extraction
import featureExtraction as fe

# classification

####################################################################################
import util

command_recordings_dir = Path("D:\Programmi\PyCharm Projects\SpeechRecognition\Comandi Samu\(1000,10)_ComandiDatasetSelezionati")
noise_recordings_dir = Path("D:\Programmi\PyCharm Projects\SpeechRecognition\Comandi Samu\_background_noise")
dataset_directory = Path("D:\Programmi\PyCharm Projects\SpeechRecognition")
noise_directory = Path("D:\Programmi\PyCharm Projects\SpeechRecognition")
features_directory = Path("D:\Programmi\PyCharm Projects\SpeechRecognition")

# Fixed sample rate of the audio signals analyzed
sample_rate = 16000  # Hz Sampling frequency of this particular dataset

dataset_name = "myData.p"
noise_name = "myNoise.p"

# Load data:
print("Loading the dataset...")
raw_dataset = ds.load_dataset(dataset_directory, command_recordings_dir, dataset_name=dataset_name)
print("Dataset loaded!")

# print("Loading noise signals...")
# noise_signals = ds.load_noise(noise_directory, noise_recordings_dir, noise_name=noise_name)
# print("Noise signals loaded!")

print("Pre-processing data...")
fine_dataset = sp.pre_processing(raw_dataset, sample_rate)
segmented_dataset = sp.segmentation(fine_dataset, sample_rate)
print("Pre-processing done!")

# //////////////
# filtered_dataset, normalized_dataset, fine_dataset = sp.new_pre_processing(raw_dataset, sample_rate)
# //////////////

mfcc_features = fe.load_mfcc_features(fine_dataset, sample_rate, features_directory, dataset_name, fixed=True)
# mfcc_features = fe.load_mfcc_features(segmented_dataset, sample_rate, features_directory, dataset_name, fixed=False)
# mod 1 builds the feature vectors computing the MFCC of every signal (variable size -> has to be fixed)

rtd_features = fe.load_rtd_features(segmented_dataset, features_directory, dataset_name)
# mod 2 builds the feature vectors computing the RTD of every signal (fixed size)

# //////////////
# dataset_spectrograms = rtd_features[0]
# dataset_features = rtd_features[1]
# //////////////


# print("Loading the train/test sets...")
# train_dataset, test_dataset = fe.loadSubsets(features, features_directory)
# print("Train/test sets loaded!")


####################################################################################

# Data Visualization:
# util.plot_class(4, raw_dataset)
# util.plot_class(4, fine_dataset)
# util.plot_class(4, mfcc_features)
# util.plot_class(4, rtd_features)


# i = 7
# util.plot_class(i, raw_dataset)
# util.plot_class(i, filtered_dataset)
# util.plot_class(i, normalized_dataset)
# util.plot_class(i, fine_dataset)
# util.plot_class(i, dataset_spectrograms)
# util.plot_class(i, dataset_features)


print("raw dataset:")
util.get_parameters(raw_dataset)

print("fine dataset:")
util.get_parameters(fine_dataset)

print("mfcc_features:")
util.get_parameters(mfcc_features)

print("rtd_features:")
util.get_parameters(rtd_features)

# util.view_example("yes23.wav", raw_dataset, mfcc_features, rtd_features)
util.view_mfcc_example("off23.wav", raw_dataset, mfcc_features)


plt.show()
