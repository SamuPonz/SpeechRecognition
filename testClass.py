
# The following file is used to test functions
import time
from pathlib import Path
from random import randrange

from playsound import playsound
from scipy.io import wavfile

import datasetMenagment as ds
import featureExtraction
import mfccMethods
import preProcessing as sp
import rtdMethods as rtd
import matplotlib.pyplot as plt
import util
import warnings


# This function interrupts the program in the case of warnings
# This is useful to quickly spot which commands give problems
warnings.showwarning = util.handle_warning


# *** TEST SECTION ***

audiofiles_directory = Path("D:\Programmi\PyCharm Projects\SpeechRecognition\Comandi Samu\(1000,10)_ComandiDatasetSelezionati")
dataset_directory = Path("D:\Programmi\PyCharm Projects\SpeechRecognition")
features_directory = Path("D:\Programmi\PyCharm Projects\SpeechRecognition")

dataset_name = "myData.p"
dataset = ds.load_dataset(dataset_directory, audiofiles_directory, dataset_name)
sample_rate = 16000

# filtered_dataset = sp.new_pre_processing(dataset, sample_rate)
# util.plot_class(5, dataset)
# util.plot_class(5, filtered_dataset)

signal = dataset.get("down1.wav")
filtered = sp.noise_reduction(signal, sample_rate)
normalized = sp.new_normalization(filtered)
segmented = sp.silence_removal(normalized, sample_rate)
plt.subplot(4, 1, 1)
plt.plot(signal)
plt.subplot(4, 1, 2)
plt.plot(filtered)
plt.subplot(4, 1, 3)
plt.plot(normalized)
plt.subplot(4, 1, 4)
plt.plot(segmented)

# #######
#
# K = 5  # K = 5
# M = 20  # M = 40
# W = 2 ** (K + 3)
# scaled = 0
# C = 0.5
#
# normalized_dataset = {k: sp.new_normalization(v) for k, v in dataset.items()}
# dataset_noise_level = sp.silence_threshold(normalized_dataset, C)
# segmented_dataset = {k: sp.segmentation(v, dataset_noise_level) for k, v in normalized_dataset.items()}
# segmented_dataset = normalized_dataset
#
# rtd_spectrograms = {k: rtd.rtd_new(v, W, scaled) for k, v in segmented_dataset.items()}
# rtd_features = {key: rtd.build_feature_vector(v, M, key) for key, v in rtd_spectrograms.items()}
#
#
# ########
#
# mfcc_features = {k: mfccMethods.mfcc_processing(v, sample_rate) for k, v in dataset.items()}
#
# ########


# FUNCTIONS USED TO STUDY THE DATASET:

# j = randrange(1000)  # select j in the range [0:999]
# j = 400
# Interesting signals found: 38, 39, 60 (this one has very low amplitude's values), 566
#
# util.view_random_rtd_example(j, dataset, normalized_dataset, segmented_dataset, rtd_spectrograms, rtd_features)
# # util.viewRandomMfccExample(j, dataset, mfcc_features)
#
# util.plot_class(8, normalized_dataset)
# print("normalized dataset:")
# util.get_parameters(normalized_dataset)
#
# util.plot_class(8, segmented_dataset)
# print("segmented dataset:")
# util.get_parameters(segmented_dataset)
#
# util.plot_class(8, rtd_spectrograms)
# print("rtd_spectrograms:")
# util.get_parameters(rtd_spectrograms)
#
# util.plot_class(8, rtd_features)
# print("rtd_features:")
# util.get_parameters(rtd_features)
#
# util.plotClass(8, mfcc_features)
# print("rtd_features:")
# util.getParameters(mfcc_features)

# util.plotDataset(segmented_dataset)
# util.castingInfluence(dataset)

plt.show()
