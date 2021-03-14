# ------------------------------------------------- Imports -----------------------------------------------------------

import warnings
from pathlib import Path
import matplotlib.pyplot as plt

# useful functions
import util

# read and import audio files
import datasetMenagment as ds

# pre-processing
import preProcessing as sp

# feature extraction
import featureExtraction as fe

# classification
import classification as clf

###############################################################

# This function interrupts the program in the case of warnings
# This is useful to quickly spot commands give problems
# warnings.showwarning = util.handle_warning

###############################################################

# ------------------------------------------------ Settings -----------------------------------------------------------

# Setting paths:
from recognitionMethods import recognition

command_recordings_dir = Path("D:\Programmi\PyCharm Projects\SpeechRecognition\Comandi Samu\(1000,10)_"
                              "ComandiDatasetSelezionati")
noise_recordings_dir = Path("D:\Programmi\PyCharm Projects\SpeechRecognition\Comandi Samu\_background_noise")
dataset_directory = Path("D:\Programmi\PyCharm Projects\SpeechRecognition")
noise_directory = Path("D:\Programmi\PyCharm Projects\SpeechRecognition")
features_directory = Path("D:\Programmi\PyCharm Projects\SpeechRecognition")
images_directory = Path("D:\Programmi\PyCharm Projects\SpeechRecognition\Images")

# Parameters:

# Fixed sample rate of the audio signals analyzed
sample_rate = 16000  # Hz Sampling frequency of this particular dataset
# 16 kHz is a good sampling frequency since the useful spectral information of the voice signal is not grater than 8 kHz

# Custom name of the dataset, this sets the name of all the further processed datasets
dataset_name = "datasetSamu.p"

# Custom name of the dataset of background noise signals
noise_name = "myNoise.p"

# ------------------------------------------------- Dataset -----------------------------------------------------------

# The dataset is a dictionary built from a directory containing sub-directories with samples of voice commands, each
# sub-dir named as the voice command class (this is necessary also for creating the "labels" list).
# The choice of a dictionary is done in order to give a name to every sample in the dataset, this turned out to be very
# useful when debugging the code in every single step of the machine learning algorithm.

print("Loading the stored dataset...")
raw_dataset, labels = ds.load_dataset(dataset_directory, command_recordings_dir, dataset_name=dataset_name)
print("Dataset loaded!")

# Load noise signals (treated as the voice commands samples)
# print("Loading noise signals...")
# noise_signals = ds.load_noise(noise_directory, noise_recordings_dir, noise_name=noise_name)
# print("Noise signals loaded!")

# --------------------------------------------- Pre processing --------------------------------------------------------

# Pre processing: filtering and normalization (the latter is a particular normalization necessary for the RDT)
print("Loading stored pre-processing data...")
fine_dataset = sp.load_preprocessed_dataset(raw_dataset, sample_rate, dataset_directory, dataset_name)
print("Pre-processed data loaded!")

# Segmentation: this is a simple method to trim the silence parts in the samples, useless information
print("Loading stored segmented data...")
segmented_dataset = sp.load_segmented_dataset(fine_dataset, sample_rate, dataset_directory, dataset_name)
# The following is used in case it is wanted to perform segmentation on unprocessed signals
# segmented_dataset = sp.load_segmented_dataset(raw_dataset, sample_rate, dataset_directory, dataset_name)
print("Segmented data loaded!")


# Librosa Dataset stretching:
# print("Loading stored stretched data...")
# print("suppressed in order to test the classifier rapidly")
# stretched_dataset = sp.load_stretched_dataset(segmented_dataset, dataset_directory, dataset_name)
# print("Stretched dataset loaded!")

# Verify the effect of the stretch on particular signals
# util.reproduce_audio("down0", stretched_dataset)

# -------------------------------------------- Feature extraction ----------------------------------------------------

# Mel Cepstrum Frequency Coefficients (MFCC): 13 static and 26 dynamic MFCC coefficients are extracted from windows of
# the signals, these are used as features in the classification stage.
print("Loading stored mfcc features...")
# Working on entire signals, on windows of different length, producing a fixed dimension feature matrix
mfcc_features = fe.load_mfcc_features(fine_dataset, sample_rate, features_directory, dataset_name, fixed=False)
# Working on segmented signals, on windows of fixed length, producing a variable dimension feature matrix (the number
# of columns changes, depending on the length of the signals
# mfcc_features = fe.load_mfcc_features(segmented_dataset, sample_rate, features_directory, dataset_name, fixed=False)
print("Mfcc features loaded!")

# Reaction Diffusion Transform (RDT): a fixed number of clustering coefficients is extracted from fixed-length windows
# of the signals. The clustering coefficients are scalars that give information regarding the frequency content of the
# signals.
print("Loading stored rtd features...")
rdt_features = fe.load_rdt_features(segmented_dataset, features_directory, dataset_name)
# The following is used in case it is wanted to perform feature extraction on the stretched data
# rdt_features = fe.load_rdt_features(stretched_dataset, features_directory, dataset_name)
print("Rtd features loaded!")

# rdt_features, dataset_spectrograms = fe.rdt_method(segmented_dataset)


# ---------------------------------------------- Classification -------------------------------------------------------

print("Creation of a model for rtd features recognition...")
clf.rtd_classification(rdt_features, labels)

# print("Classification of mfcc features...")
# clf.mfcc_classification(mfcc_features, labels)

# --------------------------------------------- Data Visualization: ---------------------------------------------------

plt.ioff()

# Plot all classes:

# for label in labels:
#     raw_image_name = "raw " + label + ".png"
#     segmented_image_name = "segmented " + label
#     util.plot_class(label, raw_dataset)
#     plt.savefig(images_directory / raw_image_name)
#     plt.close()
#     util.plot_class(label, segmented_dataset)
#     plt.savefig(images_directory / segmented_image_name)
#     plt.close()
#     util.plot_class(label, rdt_features)

# Plot specific classes
# util.plot_class("on", rdt_features)
# util.plot_class("on", raw_dataset)
# util.plot_class("on", fine_dataset)
# util.plot_class("on", segmented_dataset)
# util.plot_class("on", stretched_dataset)


# Print parameters:

#
# print("Data dimensions:")
#
# print("raw dataset:")
# util.get_parameters(raw_dataset)
#
# print("fine dataset:")
# util.get_parameters(fine_dataset)
#
# print("segmented dataset:")
# util.get_parameters(segmented_dataset)
#
# # print("stretched dataset:")
# # util.get_parameters(stretched_dataset)
#
# print("mfcc_features:")
# util.get_parameters(mfcc_features)
#
# # print("rtd_spectrograms:")
# # util.get_parameters(dataset_spectrograms)
#
# print("rtd_features:")
# util.get_parameters(rdt_features)


# ------------------------------------------------- Fast operations ---------------------------------------------------

# Computation of the overall number of samples in the datasets

# raw_data = 0
# seg_data = 0
# mfcc_data = 0
# rdt_data = 0
#
# for sig in segmented_dataset:
#     raw_data += raw_dataset[sig].size
#     seg_data += segmented_dataset[sig].size
#     mfcc_data += mfcc_features[sig].size
#     rdt_data += rdt_features[sig].size
#
# print(raw_data)
# print(seg_data)
# print(mfcc_data)
# print(rdt_data)


# Reproduce an example of an audio file after every operation:

# util.reproduce_raw_audio("off0", raw_dataset)
# util.reproduce_normalized_audio("off0", fine_dataset, 16000)
# util.reproduce_normalized_audio("off0", segmented_dataset, 16000)
# util.reproduce_normalized_audio("down0", stretched_dataset, 16000)


# Comparison of different rtd features of the same signal

# import rdtMethods as rdt
# name = "go14"
# signal = segmented_dataset[name]
# rdt1 = rdt.rdt_new(name, signal, 2**(1 + 3), 0)
# rdt2 = rdt.rdt_new(name, signal, 2**(3 + 3), 0)
# rdt3 = rdt.rdt_new(name, signal, 2**(6 + 3), 0)
# rdt4 = rdt.rdt_new(name, signal, 2**(9 + 3), 0)

# -------------------------------------------------- Examples ---------------------------------------------------------

# util.view_example("down71", raw_dataset, segmented_dataset, mfcc_features, rdt_features)
# util.view_example("no63", raw_dataset, segmented_dataset, rdtnot, rdtsca)
# util.plottoez("down71", segmented_dataset)
# util.plottoez2("down71", signal, rdt1, rdt2, rdt3, rdt4)

# util.view_mfcc_example("off23.wav", raw_dataset, mfcc_features)
# util.parameter_examples()

# ------------------------------------------------- Recognition -------------------------------------------------------

print("Let's try to recognise something!")
while True:
    recognition(command_recordings_dir)
    if input("Repeat the program? (Y/N)").strip().upper() != 'Y':
        break

plt.show()
