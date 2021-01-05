
# The following file is used to test functions

from pathlib import Path
from random import randrange

import datasetMenagment as ds
import featureExtraction
import mfccMethods
import signalPreprocessing as sp
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
dataset = ds.loadDataset(dataset_directory, audiofiles_directory,  dataset_name)

#######

K = 5  # K = 5
M = 20  # M = 40
W = 2 ** (K + 3)
scaled = 0
C = 0.5

normalized_dataset = {k: sp.newNormalization(v) for k, v in dataset.items()}
dataset_noise_level = sp.silenceThreshold(normalized_dataset, C)
segmented_dataset = {k: sp.segmentation(v, dataset_noise_level) for k, v in normalized_dataset.items()}
segmented_dataset = normalized_dataset

rtd_spectrograms = {k: rtd.rtdNew(v, W, scaled) for k, v in segmented_dataset.items()}
rtd_features = {key: rtd.buildFeatureVector(v, M, key) for key, v in rtd_spectrograms.items()}

########

sample_rate = 16000

mfcc_features = {k: mfccMethods.mfccProcessing(v, sample_rate) for k, v in dataset.items()}

########

# FUNCTIONS USED TO STUDY THE DATASET:

j = randrange(1000)  # select j in the range [0:999]
j = 400
# Interesting signals found: 38, 39, 60 (this one has very low amplitude's values), 566

util.viewRandomRtdExample(j, dataset, normalized_dataset, segmented_dataset, rtd_spectrograms, rtd_features)
# util.viewRandomMfccExample(j, dataset, mfcc_features)

util.plotClass(8, normalized_dataset)
print("normalized dataset:")
util.getParameters(normalized_dataset)

util.plotClass(8, segmented_dataset)
print("segmented dataset:")
util.getParameters(segmented_dataset)

util.plotClass(8, rtd_spectrograms)
print("rtd_spectrograms:")
util.getParameters(rtd_spectrograms)

util.plotClass(8, rtd_features)
print("rtd_features:")
util.getParameters(rtd_features)
#
# util.plotClass(8, mfcc_features)
# print("rtd_features:")
# util.getParameters(mfcc_features)

# util.plotDataset(segmented_dataset)
# util.castingInfluence(dataset)

plt.show()
