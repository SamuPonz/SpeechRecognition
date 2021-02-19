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
warnings.showwarning = util.handle_warning

###############################################################

# ------------------------------------------------ Settings -----------------------------------------------------------

# Setting paths:
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

# Dataset stretching:
# This is a simple interpolation of the signals, used to increment the number of samples of particularly short signals
# and increase performance.
# This operation does not preserve the frequency content of the signals (it changes the pitch) and sadly it does not
# increase the performance of the algorithm. It was to good to be true!

# print("stretching the data...")
# minimum_length = 8192+4096
# stretched_dataset = sp.stretching_correction(segmented_dataset, minimum_length)
# print("Data stretched!")

# Verify the effect of the stretch on particular signals
# util.reproduce_audio("down0", stretched_dataset)

# -------------------------------------------- Feature extraction ----------------------------------------------------

# Mel Cepstrum Frequency Coefficients (MFCC): 13 static and 26 dynamic MFCC coefficients are extracted from windows of
# the signals, these are used as features in the classification stage.
print("Loading stored mfcc features...")
# Working on entire signals, on windows of different length, producing a fixed dimension feature matrix
mfcc_features = fe.load_mfcc_features(fine_dataset, sample_rate, features_directory, dataset_name, fixed=True)
# Working on segmented signals, on windows of fixed length, producing a variable dimension feature matrix (the number
# of columns changes, depending on the length of the signals
# mfcc_features = fe.load_mfcc_features(segmented_dataset, sample_rate, features_directory, dataset_name, fixed=False)
print("Mfcc features loaded!")

# Reaction Diffusion Transform (RDT): a fixed number of clustering coefficients is extracted from fixed-length windows
# of the signals. The clustering coefficients are scalars that give informations regarding the frequency content of the
# signals.
print("Loading stored rtd features...")
rdt_features = fe.load_rdt_features(segmented_dataset, features_directory, dataset_name)
# The following is used in case it is wanted to perform feature extraction on the stretched data
# rdt_features = fe.load_rdt_features(stretched_dataset, features_directory, dataset_name)
print("Rtd features loaded!")


# ---------------------------------------------- Classification -------------------------------------------------------

print("Classification of rtd features...")
clf.rtd_classification(rdt_features, labels)

# print("Classification of mfcc features...")
# clf.mfcc_classification(mfcc_features, labels)

# --------------------------------------------- Data Visualization: ---------------------------------------------------

plt.ioff()

# Plot all classes:

for label in labels:
    raw_image_name = "raw " + label + ".png"
    segmented_image_name = "segmented " + label

    # util.plot_class(label, raw_dataset)
    # plt.savefig(images_directory / raw_image_name)
    # plt.close()
    # util.plot_class(label, segmented_dataset)
    # plt.savefig(images_directory / segmented_image_name)
    # plt.close()
    # util.plot_class(label, rdt_features)

# Plot specific classes
# util.plot_class("go", rdt_features)


# Print parameters:

print("Data dimensions:")
print()

print("raw dataset:")
util.get_parameters(raw_dataset)

print("fine dataset:")
util.get_parameters(fine_dataset)

print("segmented dataset:")
util.get_parameters(segmented_dataset)

# print("segmented dataset:")
# util.get_parameters(stretched_dataset)

# print("mfcc_features:")
# util.get_parameters(mfcc_features)

print("rtd_features:")
util.get_parameters(rdt_features)

# -------------------------------------------------- Examples ---------------------------------------------------------

# util.view_example("up0", raw_dataset, segmented_dataset, mfcc_features, rdt_features)

# util.view_mfcc_example("off23.wav", raw_dataset, mfcc_features)

plt.show()
