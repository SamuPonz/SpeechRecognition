from _library_Mfccs import mfcc, delta
import rtdMethods as rtd
import numpy as np
import fileManagment as fm
import pandas as pd
from sklearn.model_selection import train_test_split

from util import measure


@measure
def loadFeatures(dataset, sample_rate, features_directory, dataset_name, method=2):
    features_name = ""
    if method == 1:
        features_name = 'mfccs_features_' + dataset_name
    elif method == 2:
        features_name = 'rdt_features_' + dataset_name

    features_file = features_directory / features_name

    if features_file.is_file():
        print("Loading stored features...")
        return fm.readFile(features_file)
    else:
        print("No saved features found. New feature extraction...")
        features, dataset_noise_level = buildFeatures(dataset, sample_rate, method)
        fm.createFile(features_name, [features, dataset_noise_level])
        return features, dataset_noise_level


def buildFeatures(dataset, sample_rate, method):
    if method == 1:
        dataset_noise_level = 0
        features = mfccMethod(dataset, sample_rate)
        return features, dataset_noise_level

    elif method == 2:
        features, dataset_noise_level = rtdMethod(dataset)
        return features, dataset_noise_level


def rtdMethod(dataset):
    K = 4
    M = 7
    scaled = 0  # if this flag is = 0 classic RTD is computed, if it is = 1 scaled RTD is computed
    C = 0.5  # used to set the silence threshold

    # M and K are two predetermined parameters that have to be optimized manually looking at the performance of the
    # entire classification process.

    # K is the number of channels of the RTD, it is the number of rows of the feature vectors.
    # K = floor(log2(W))-3, it depends on W (window size applied to the audio signal), which ranges in 16 <= W <= N.
    # where N is the number of samples of the signal. To be sure that this condition is respected, the N value to use
    # should be the lowest of the dataset of audio signals.
    # LOWER BOUND OF K: K = 1, when W = 16.
    # This case implies that the feature vector is a row vector, where each value is the clustering coefficient that
    # represents the frequency content of that window of the signal.
    # UPPER BOUND OF K: K = floor(log2(N))-3, when W = N.
    # floor(log2(N))-3 is the maximum "frequency resolution" (n. of rows of the spectrograms) that can be obtained.
    # In this case, only one spectrum will be computed with the RTD, segmentation of the spectrogram is not needed
    # (M = 1).
    # The feature vector will be only a column signal, we will have floor(log2(N))-3 clustering coefficients
    # representing the spectral content of the whole signal.

    # M = 7 is the number of segments of the spectrogram, is the number of columns of the feature vector matrix.
    # M is used to fix the number of columns (T) of the spectrograms, by averaging every Ts column, value that changes
    # for every signal spectrum because Ts = T//M. M <= T. the T value to use should be the lowest of the dataset of
    # spectrograms. Since T = N//M, the N value to use should be the lowest of the dataset of the audio signals.
    # LOWER BOUND OF M: M = 1.
    # In this case all the columns of the spectrograms are averaged in one single column i.e. one single spectrum.
    # M is set to 1 if we have a single spectrum and not a spectrogram (the case in which W = N s.t.
    # K = floor(log2(N))-3).
    # UPPER BOUND OF M: M = T.
    # In this case no average is performed, T = N//W, so N//16 is the maximum value (the case in which W = 16 s.t.
    # K = 1)

    # SUMMARIZING:
    # The value of W (window-size) is automatically limited to the interval 16 <= W <= N by the mathematical rules of
    # RTD, if W = N, K = floor(log2(N))-3; if W = 16, K = 1.
    # The value of K bounds the value of M to the interval 1 <= M <= N//W = N//(2**(K+3));
    # The value of M does not limit the value of K or W.
    # In order to be sure that M and K are possible values, choose them looking at the minimum number of samples of all
    # the audio signals.

    # EXAMPLES:
    # 1) Let's take N = 20'480 (if the sampling frequency is 16kHz this is an audio signal of 1.28 seconds)
    # if W 1024 = 2**(7+3), K = 7 and if we set M = 5 the dimensions of the feature vectors are [K, M] = [7,5]
    # Having 1024 samples for every window, we will have 20 windows that will be transformed to M = 5 segments,
    # averaging on every 4 windows (i.e. the columns of the spectrum).

    # 2) Let's take N = 1017 (This is the minimum length of the signals of the dataset after segmentation with C = 0.5)
    # Let's chose K and M respecting the relative upper bounds of these values.
    # K <= floor(log2(N))-3:
    # This means W must be <= than N, e.g. W = 128, and so K = 4 (condition to W and thus K is respected).
    # M <= N//W:
    # N//W = 1017//128 = 7 (this is the minimum number of columns of the spectrogram's dataset), since 7 is already low
    # (due to the silence-segmentation stage) we can set M = 7.
    # If W 128 = 2**(4+3), K = 4 and if we set M = 7, the dimensions of the feature vectors will be [K, M] = [4,7].

    # Limit cases:
    # 1) if W = 16, K = 1. Therefore:  1 <= M <= N//16 (20'480//16 = 1280 windows!)
    # If for instance we take once again M = 5, the dimension of the feature vectors will be  [K, M] = [1,5] and the
    # 1280 windows will be converted to 5 segments, averaging on every 256 windows (i.e. the columns). In this case
    # would be better to set M at higher values to maintain the time resolution that we have privileged over the
    # spectral resolution.
    # 2) if W = N, K = floor(log2(N))-3 =  We only have 1 window, therefore M = 1.

    W = 2 ** (K + 3)  # window size, delay in the RTD
    # K = log2(W)-3

    # The Dataset is normalized: all the signals are rescaled in the range [-1,1] in order to apply the RTD.
    # Indeed this is one of the assumptions about the signal on which it applied the RTD.
    normalized_dataset = {k: rtd.newNormalization(v) for k, v in dataset.items()}

    # It is useful do to apply a filter that can be seen as an "economic" segmentation: in order to reduce the samples
    # of the signals all the samples whose absolute value is below a particular threshold are discarded. It is crucial
    # to select the threshold properly. Since in the references was not specified the method usd to set the threshold,
    # what it is done here is a personal solution. In this method, it is simply computed the mean on the first n
    # elements of every signal. The means are collected in a vector and the threshold is set to E[means] + C*std[means],
    # where C is a particular coverage factor which value is set manually, depending on the dataset.
    # C = 0.5

    dataset_noise_level = rtd.silenceThreshold(normalized_dataset, C)

    segmented_dataset = {k: rtd.segmentation(v, dataset_noise_level) for k, v in normalized_dataset.items()}
    # The following line is used to ignore the segmentation stage
    # segmented_dataset = normalized_dataset

    # Different trials were made in order to understand what was the best order in which performing these two steps:
    # 1) segmentation -> normalization: due to the high dynamic range between the absolute maximum amplitudes of the
    # signals in the dataset, and due also to the questionable method used to compute the silence threshold (which looks
    # at the first n samples of every signal, where it is not ensured that there is silence), with a relatively high
    # threshold's value some signals may be completely deleted. This causes some errors in the subsequent sections of
    # the program. If normalization is performed before, all signals have the same max and min values! The amplitude
    # information is completely lost?

    # 2) normalization -> segmentation: In this way the problem stated above is partially solved, but other problems
    # arise: at first it was selected a standard technique for the normalization, where the signal is normalized in the
    # interval [0,1], than scaled by 2 (hence in [0,2]) and finally shifted down by one unit (hence [-1,1]). With this
    # method, the baseline of the signal was different in every single signal, this effect occurs if the range of the
    # original signal is not symmetrical.

    # In order to solve both problems, it was chosen the following order: normalization -> segmentation, with an
    # alternative method for the normalization. With this new method, the baseline remains at zero even after the the
    # rescaling, but the signal amplitudes are significantly changed. Have to verify if this corrupts the
    # data.
    # (I think that this is not a problem as long as this do not compromise the recognition phase: it is a
    # classification problem, not a compression one, i.e. it is not needed to reconstruct the original signal. Indeed
    # the RTD itself cannot be reversed, there is no such thing as an inverse RTD transform.

    # The RTD_new algorithm is performed in order to build the spectrograms of the signals, seems to work properly:
    # W is the window size
    dataset_spectrograms = {k: rtd.rtdNew(v, W, scaled) for k, v in segmented_dataset.items()}

    # This function builds the the fixed sized feature vector starting from the spectrograms, which can be different in
    # size (different number of columns, this represents the time dimension of the spectrograms that depends on the
    # number of samples of the signals. In order to have a fixed number of columns a number is selected (M) and the
    # averages of groups of columns is performed in order to obtain M columns in every spectrogram.
    #
    # Problems:
    # 1) Some segmented signals have very few columns, if M is greater than the number of column of a spectrgram
    # meaningless values are  introduced in the feature vector. If the number of columns is really high, the averaging
    # causes a drastic compression of the information.
    # 2) A poor segmentation that leaves in the signals the "silence intervals" causes the spectrum to have meaningful
    # information only in a limited part of it (like in the original signal). Hence it this likely that all the
    # information that resides in some columns is averaged (i.e. compressed) in one single columns, leaving a lot of
    # columns without meaningful information.
    #
    # A solution could be to do a smart averaging, selecting the columns (i.e. spectrums) where the values are really
    # low, leaving intact the high values spectrums, This is not so convincing because it causes a "distortion" of the
    # time axis, more that the actual averaging method. Actually, thinking about it, the time distortion it is already
    # introduced by the segmentation stage itself...
    dataset_features = {key: rtd.buildFeatureVector(v, M, key) for key, v in dataset_spectrograms.items()}

    return dataset_features, dataset_noise_level


def mfccMethod(dataset, sample_rate):
    features = {k: mfccProcessing(v, sample_rate) for k, v in dataset.items()}
    return features


def dataSplit(features, train_size=0.7):
    # Creating testing and training sets

    s = pd.Series(features)
    training_dataset, test_dataset = [i.to_dict() for i in train_test_split(s, train_size=train_size)]
    return training_dataset, test_dataset


def loadSubsets(features, features_directory):
    # Loads saved train and test sets, creates them if there are none
    train_name = "train_features.p"
    test_name = "test_features.p"
    train_file = features_directory / train_name
    test_file = features_directory / test_name
    if not train_file.is_file() or not test_file.is_file():
        print("No saved train/test sets found. Building two new random sets...")
        train_dataset, test_dataset = dataSplit(features, train_size=0.7)
        fm.createFile(train_name, train_dataset)
        fm.createFile(test_name, test_dataset)
    return fm.readFile(train_file), fm.readFile(test_file)


# The following method should be placed in another file
# It is placed here in order to not modify the library file "_library_Mfccs.py"
def mfccProcessing(signal, sample_rate):
    mfcc_feat = mfcc(signal=signal, samplerate=sample_rate, winlen=0.025, winstep=0.01, numcep=13,
                     nfilt=26, nfft=512, lowfreq=300, highfreq=None, preemph=0.95, ceplifter=0,
                     appendEnergy=True, winfunc=np.hamming)
    d_mfcc_feat = delta(mfcc_feat, 2)
    dd_mfcc_feat = delta(d_mfcc_feat, 2)
    feature_vector = np.concatenate((mfcc_feat, d_mfcc_feat, dd_mfcc_feat), axis=1)
    return feature_vector
