
# The following file is used to test functions
from pathlib import Path

import librosa
import scipy
import datasetMenagment as ds
import preProcessing as sp
import util
import warnings
import matplotlib.pyplot as plt


# This function interrupts the program in the case of warnings
# This is useful to quickly spot which commands give problems
warnings.showwarning = util.handle_warning


# *** TEST SECTION ***

audiofiles_directory = Path("D:\Programmi\PyCharm Projects\SpeechRecognition\Comandi Samu\(1000,10)_ComandiDatasetSelezionati")
dataset_directory = Path("D:\Programmi\PyCharm Projects\SpeechRecognition")
features_directory = Path("D:\Programmi\PyCharm Projects\SpeechRecognition")

dataset_name = "datasetSamu.p"
dataset, labels = ds.load_dataset(dataset_directory, audiofiles_directory, dataset_name)
sample_rate = 16000

# filtered_dataset = sp.new_pre_processing(dataset, sample_rate)
# util.plot_class(5, dataset)
# util.plot_class(5, filtered_dataset)

name = "down1"
signal = dataset[name]
filtered = sp.noise_reduction(signal, sample_rate)
normalized = sp.new_normalization(filtered)
segmented = sp.silence_removal(name, normalized, sample_rate)

# Plotting the complete pre-processing phase
# plt.subplot(4, 1, 1)
# plt.plot(signal)
# plt.subplot(4, 1, 2)
# plt.plot(filtered)
# plt.subplot(4, 1, 3)
# plt.plot(normalized)
# plt.subplot(4, 1, 4)
# plt.plot(segmented)

#############################################
#
# from scipy.signal import butter, lfilter
#
#
# def butter_bandpass(lowcut, highcut, fs, order=20):
#     nyq = 0.5 * fs
#     low = lowcut / nyq
#     high = highcut / nyq
#     b, a = butter(order, [low, high], btype='band')
#     return b, a
#
#
# def butter_bandpass_filter(data, lowcut, highcut, fs, order=20):
#     b, a = butter_bandpass(lowcut, highcut, fs, order=order)
#     y = lfilter(b, a, data)
#     return y
#
#
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.signal import freqz
#
# # Sample rate and desired cutoff frequencies (in Hz).
# fs = 16000.0
# lowcut = 300.0
# highcut = 3400.0
#
# # Plot the frequency response for a few different orders.
# plt.figure(1)
# plt.clf()
# for order in [10]:
#     b, a = butter_bandpass(lowcut, highcut, fs, order=order)
#     w, h = freqz(b, a, worN=2000)
#     plt.plot((fs * 0.5 / np.pi) * w, abs(h), label="order = %d" % order)
#
# plt.plot([0, 0.5 * fs], [np.sqrt(0.5), np.sqrt(0.5)],
#          '--', label='sqrt(0.5)')
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Gain')
# plt.grid(True)
# plt.legend(loc='best')
#
# # Filter a noisy signal.
#
# name = "off41"
# T = 1/fs
# x = dataset[name]
# t = np.linspace(0, 1, x.shape[0], endpoint=False)
# a = 0.02
# plt.figure(2)
# plt.clf()
# plt.plot(t, x, label='Noisy signal')
#
# y = butter_bandpass_filter(x, lowcut, highcut, fs, order=10)
# plt.plot(t, y, label='Filtered signal')
# plt.xlabel('time (seconds)')
# plt.hlines([-a, a], 0, T, linestyles='--')
# plt.grid(True)
# plt.axis('tight')
# plt.legend(loc='upper left')

#######################################################
# from scipy import signal
# lowcut = 300  # Hz
# highcut = 4000  # Hz
# order = 10
# nyq = 0.5 * sample_rate  # 8000 Hz
# low = lowcut / nyq  # 300/8000 = 0.0375
# high = highcut / nyq  # 3400/8000 = 0.425
#
# b, a = signal.butter(order, [low, high], analog=False, btype='band', output='ba')
# sos = signal.butter(order, [low, high], analog=False, btype='band', output='sos')
# x = signal.unit_impulse(8000)
# y_tf = signal.lfilter(b, a, x)
# y_sos = signal.sosfilt(sos, x)
# plt.grid(True)
# plt.plot(y_tf, 'r', label='TF')
# plt.plot(y_sos, 'k', label='SOS')
# plt.legend(loc='best')
# plt.show()
#######################
import matplotlib.pyplot as plt
from scipy import signal
b, a = signal.ellip(13, 0.009, 80, 0.05, output='ba')
sos = signal.ellip(13, 0.009, 80, 0.05, output='sos')
x = signal.unit_impulse(700)
y_tf = signal.lfilter(b, a, x)
y_sos = signal.sosfilt(sos, x)
plt.grid(True)
plt.plot(y_tf, 'r', label='TF')
plt.plot(y_sos, 'k', label='SOS')
plt.legend(loc='best')
plt.show()
##############################

# song, fs = librosa.load("go1.wav", sr=None)
#
# song_2_times_faster = librosa.effects.time_stretch(song, 0.2)
#
# scipy.io.wavfile.write("song_2_times_faster.wav", fs, song_2_times_faster) # save the song


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
