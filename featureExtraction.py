import rtdMethods as rtd
import mfccMethods as mfcc
import fileManagment as fm
from util import measure


@measure
def load_mfcc_features(dataset, sample_rate, features_directory, dataset_name, fixed=False):
    features_name = 'mfcc_features_' + dataset_name
    features_file = features_directory / features_name

    if features_file.is_file():
        return fm.read_file(features_file)
    else:
        print("No saved mfcc features found. New feature extraction...")
        features = mfcc_method(dataset, sample_rate, fixed)
        fm.create_file(features_name, features)
        return features


@measure
def load_rtd_features(dataset, features_directory, dataset_name):
    features_name = 'rdt_features_' + dataset_name
    features_file = features_directory / features_name

    if features_file.is_file():
        return fm.read_file(features_file)
    else:
        print("No saved rtd features found. New feature extraction...")
        features = rtd_method(dataset)
        fm.create_file(features_name, features)
        return features


def rtd_method(dataset):
    K = 4
    M = 8
    scaled = 0  # if this flag is = 0 classic RTD is computed, if it is = 1 scaled RTD is computed

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

    # M is the number of segments of the spectrogram, is the number of columns of the feature vector matrix.
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

    # The RTD_new algorithm is performed in order to build the spectrograms of the signals, seems to work properly:
    # W is the window size
    dataset_spectrograms = {k: rtd.rtd_new(v, W, scaled) for k, v in dataset.items()}

    # This function builds the the fixed sized feature vector starting from the spectrograms, which can be different in
    # size (different number of columns, this represents the time dimension of the spectrograms that depends on the
    # number of samples of the signals. In order to have a fixed number of columns a number is selected (M) and the
    # averages of groups of columns is performed in order to obtain M columns in every spectrogram.

    # Problems:
    # 1) Some segmented signals have very few columns, if M is greater than the number of column of a spectrogram
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

    dataset_features = {key: rtd.build_feature_vector(v, M, key) for key, v in dataset_spectrograms.items()}

    return dataset_features

    # //////////////
    # return dataset_spectrograms, dataset_features
    # //////////////


def mfcc_method(dataset, sample_rate, fixed):
    common_win_len = 0.025  # ms
    common_win_step = 0.01  # ms

    fixed_number_of_frames, overlap_factor = mfcc.optimal_number_of_frames(dataset, sample_rate, common_win_len,
                                                                           common_win_step)

    features = {k: mfcc.mfcc_processing(v, sample_rate, common_win_len, common_win_step, fixed_number_of_frames,
                                        overlap_factor, fixed) for k, v in dataset.items()}
    return features

