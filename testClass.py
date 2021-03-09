
# The following file is used to test functions
from pathlib import Path

import librosa
import scipy
from sklearn.datasets import make_circles
from sklearn.svm import SVC

import datasetMenagment as ds
import preProcessing as sp
import util
import warnings
import matplotlib.pyplot as plt


# This function interrupts the program in the case of warnings
# This is useful to quickly spot which commands give problems
# warnings.showwarning = util.handle_warning


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
# import matplotlib.pyplot as plt
# from scipy import signal
# b, a = signal.ellip(13, 0.009, 80, 0.05, output='ba')
# sos = signal.ellip(13, 0.009, 80, 0.05, output='sos')
# x = signal.unit_impulse(700)
# y_tf = signal.lfilter(b, a, x)
# y_sos = signal.sosfilt(sos, x)
# plt.grid(True)
# plt.plot(y_tf, 'r', label='TF')
# plt.plot(y_sos, 'k', label='SOS')
# plt.legend(loc='best')
# plt.show()
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
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn import svm
# from sklearn.datasets import make_blobs
#
#
# # we create 40 separable points
# X, y = make_blobs(n_samples=40, centers=2, random_state=6)
#
# # fit the model, don't regularize for illustration purposes
# clf = svm.SVC(kernel='linear', C=1000)
# clf.fit(X, y)
#
# plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.rainbow)
#
# # plot the decision function
# ax = plt.gca()
# xlim = ax.get_xlim()
# ylim = ax.get_ylim()
#
# # create grid to evaluate model
# xx = np.linspace(xlim[0], xlim[1], 30)
# yy = np.linspace(ylim[0], ylim[1], 30)
# YY, XX = np.meshgrid(yy, xx)
# xy = np.vstack([XX.ravel(), YY.ravel()]).T
# Z = clf.decision_function(xy).reshape(XX.shape)
#
# # plot decision boundary and margins
# ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
#            linestyles=['--', '-', '--'])
# # plot support vectors
# ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
#            linewidth=1, facecolors='none', edgecolors='k')
#
# #############
#
# from sklearn.datasets.samples_generator import make_circles
# import numpy as np
#
#
# def plot_svc_decision_function(model, ax=None, plot_support=True):
#     """Plot the decision function for a 2D SVC"""
#     if ax is None:
#         ax = plt.gca()
#     xlim = ax.get_xlim()
#     ylim = ax.get_ylim()
#
#     # create grid to evaluate model
#     x = np.linspace(xlim[0], xlim[1], 30)
#     y = np.linspace(ylim[0], ylim[1], 30)
#     Y, X = np.meshgrid(y, x)
#     xy = np.vstack([X.ravel(), Y.ravel()]).T
#     P = model.decision_function(xy).reshape(X.shape)
#
#     # plot decision boundary and margins
#     ax.contour(X, Y, P, colors='k',
#                levels=[-1, 0, 1], alpha=0.5,
#                linestyles=['--', '-', '--'])
#
#     # plot support vectors
#     if plot_support:
#         ax.scatter(model.support_vectors_[:, 0],
#                    model.support_vectors_[:, 1],
#                    s=300, linewidth=1, facecolors='none')
#     ax.set_xlim(xlim)
#     ax.set_ylim(ylim)
#
#
# X, y = make_circles(500, factor=.1, noise=.1, random_state=40)
#
# clf = SVC(kernel='linear').fit(X, y)
# # plot_svc_decision_function(clf, plot_support=False)
#
# r = np.exp(-(X ** 2).sum(1))
# fig, ax = plt.subplots(1, 1)
# ax.grid(False)
# plt.scatter(X[:, 0], X[:, 1], c=y, s=25, cmap=plt.cm.rainbow)
#
# ax.set_facecolor('#f8f8f8')
#
# # clf = SVC(kernel='linear').fit(X, y)
# # plot_svc_decision_function(clf, plot_support=False)
#
# def plot_3D(elev=30, azim=30, X=X, y=y):
#     ax = plt.subplot(projection='3d')
#     ax.grid(False)
#     ax.scatter3D(X[:, 0], X[:, 1], r, c=y, s=25, cmap=plt.cm.rainbow)
#     ax.view_init(elev=elev, azim=azim)
#
#
# def f(x, y):
#     return x**2 + y**2
#
# fig = plt.figure()
#
# ax = fig.add_subplot(111, projection='3d')
# ax.grid(False)
# ax.scatter3D(X[:, 0], X[:, 1], r, c=y, s=25, cmap=plt.cm.rainbow)
# ax.view_init(elev=30, azim=30)
#
#
# x = y = np.arange(-1.0, 1.0, .1)
# X, Y = np.meshgrid(x, y)
#
# Z = f(X,Y)
#
# # ax.plot_surface(X, Y, Z,color='gray',alpha=.8)
#
#
# #To plot the surface at 100, use your same grid but make all your numbers zero
# Z2 = Z*0.+0.75
# ax.plot_surface(X, Y, Z2,color='g',alpha=.3) #plot the surface

####################################################

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV


# Utility function to move the midpoint of a colormap to be around
# the values of interest.

class MidpointNormalize(Normalize):

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

# #############################################################################
# Load and prepare data set
#
# dataset for grid search

iris = load_iris()
X = iris.data
y = iris.target

# Dataset for decision function visualization: we only keep the first two
# features in X and sub-sample the dataset to keep only 2 classes and
# make it a binary classification problem.

X_2d = X[:, :2]
X_2d = X_2d[y > 0]
y_2d = y[y > 0]
y_2d -= 1

# It is usually a good idea to scale the data for SVM training.
# We are cheating a bit in this example in scaling all of the data,
# instead of fitting the transformation on the training set and
# just applying it on the test set.

scaler = StandardScaler()
X = scaler.fit_transform(X)
X_2d = scaler.fit_transform(X_2d)

# #############################################################################
# Train classifiers
#
# For an initial search, a logarithmic grid with basis
# 10 is often helpful. Using a basis of 2, a finer
# tuning can be achieved but at a much higher cost.

C_range = np.logspace(-2, 10, 13)
gamma_range = np.logspace(-9, 3, 13)
param_grid = dict(gamma=gamma_range, C=C_range)
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
grid.fit(X, y)

print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))

# Now we need to fit a classifier for all parameters in the 2d version
# (we use a smaller set of parameters here because it takes a while to train)

C_2d_range = [1e-2, 1, 1e2]
gamma_2d_range = [1e-1, 1, 1e1]
classifiers = []
for C in C_2d_range:
    for gamma in gamma_2d_range:
        clf = SVC(C=C, gamma=gamma)
        clf.fit(X_2d, y_2d)
        classifiers.append((C, gamma, clf))

# #############################################################################
# Visualization
#
# draw visualization of parameter effects

plt.figure(figsize=(8, 6))
xx, yy = np.meshgrid(np.linspace(-3, 3, 200), np.linspace(-3, 3, 200))
for (k, (C, gamma, clf)) in enumerate(classifiers):
    # evaluate decision function in a grid
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # visualize decision function for these parameters
    plt.subplot(len(C_2d_range), len(gamma_2d_range), k + 1)
    plt.title("gamma=10^%d, C=10^%d" % (np.log10(gamma), np.log10(C)),
              size='medium')

    # visualize parameter's effect on decision function
    plt.pcolormesh(xx, yy, -Z, cmap=plt.cm.RdBu, shading='nearest')
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_2d, cmap=plt.cm.rainbow,
                edgecolors='k')
    plt.xticks(())
    plt.yticks(())
    plt.axis('tight')

scores = grid.cv_results_['mean_test_score'].reshape(len(C_range),
                                                     len(gamma_range))


####################################################
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
