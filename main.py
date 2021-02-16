# first script:
import os
from pathlib import Path
import matplotlib.pyplot as plt
import time

import util

####################################################################################

# read and import audio files
import datasetMenagment as ds

# pre-processing
import preProcessing as sp

# feature extraction
import featureExtraction as fe

# classification
import classification as clf

####################################################################################

command_recordings_dir = Path("D:\Programmi\PyCharm Projects\SpeechRecognition\Comandi Samu\(1000,10)_ComandiDatasetSelezionati")
noise_recordings_dir = Path("D:\Programmi\PyCharm Projects\SpeechRecognition\Comandi Samu\_background_noise")
dataset_directory = Path("D:\Programmi\PyCharm Projects\SpeechRecognition")
noise_directory = Path("D:\Programmi\PyCharm Projects\SpeechRecognition")
features_directory = Path("D:\Programmi\PyCharm Projects\SpeechRecognition")
images_directory = Path("D:\Programmi\PyCharm Projects\SpeechRecognition\Images")
# Fixed sample rate of the audio signals analyzed
sample_rate = 16000  # Hz Sampling frequency of this particular dataset

dataset_name = "datasetSamu.p"
noise_name = "myNoise.p"

labels = ["down", "go", "left", "no", "off", "on", "right", "stop", "up", "yes"]

# Load data:
print("Loading the stored dataset...")
raw_dataset = ds.load_dataset(dataset_directory, command_recordings_dir, dataset_name=dataset_name)
print("Dataset loaded!")

# print("Loading noise signals...")
# noise_signals = ds.load_noise(noise_directory, noise_recordings_dir, noise_name=noise_name)
# print("Noise signals loaded!")

print("Loading stored pre-processing data...")
fine_dataset = sp.load_preprocessed_dataset(raw_dataset, sample_rate, dataset_directory, dataset_name)
print("Pre-processed data loaded!")

print("Loading stored segmented data...")
segmented_dataset = sp.load_segmented_dataset(fine_dataset, sample_rate, dataset_directory, dataset_name)
print("Segmented data loaded!")

# //////////////
# filtered_dataset, normalized_dataset, fine_dataset = sp.new_pre_processing(raw_dataset, sample_rate)
# //////////////

print("Loading stored mfcc features...")
mfcc_features = fe.load_mfcc_features(fine_dataset, sample_rate, features_directory, dataset_name, fixed=True)
print("Mfcc features loaded!")
# mfcc_features = fe.load_mfcc_features(segmented_dataset, sample_rate, features_directory, dataset_name, fixed=False)
# mod 1 builds the feature vectors computing the MFCC of every signal (variable or fixed size)

print("Loading stored rtd features...")
rdt_features = fe.load_rdt_features(segmented_dataset, features_directory, dataset_name)
print("Rtd features loaded!")
# mod 2 builds the feature vectors computing the RTD of every signal (fixed size)

# //////////////
# dataset_spectrograms = rtd_features[0]
# dataset_features = rtd_features[1]
# //////////////

# print("Loading the train/test sets...")
# train_dataset, test_dataset = fe.loadSubsets(features, features_directory)
# print("Train/test sets loaded!")

####################################################################################

# Rtd Features only have to be transposed
rdt_features_T = {k: v.transpose() for k, v in rdt_features.items()}

# clf.classification_method(labels, rdt_features_T)
clf.classification_method(labels, mfcc_features)

####################################################################################

plt.ioff()

# Data Visualization:

for label in labels:
    raw_image_name = "raw " + label + ".png"
    segmented_image_name = "segmented " + label

    # util.plot_class(label, raw_dataset)
    # plt.savefig(images_directory / raw_image_name)
    # plt.close()
    # util.plot_class(label, segmented_dataset)
    # plt.savefig(images_directory / segmented_image_name)
    # plt.close()

    # util.plot_class(label, rtd_features)


# Print parameters

print("raw dataset:")
util.get_parameters(raw_dataset)

print("fine dataset:")
util.get_parameters(fine_dataset)

print("segmented dataset:")
util.get_parameters(segmented_dataset)

print("mfcc_features:")
util.get_parameters(mfcc_features)

print("rtd_features:")
util.get_parameters(rdt_features)
#
#
# Exemples:

util.view_example("on61", raw_dataset, segmented_dataset, mfcc_features, rdt_features)

# util.view_mfcc_example("off23.wav", raw_dataset, mfcc_features)


plt.show()
