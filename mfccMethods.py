from _library_Mfccs import mfcc, delta
import numpy as np


def mfcc_processing(signal, sample_rate, common_win_step, fixed_number_of_frames, K, r, fixed):
    """
    Compute the feature vector with static and dynamic features
    :param fixed_number_of_frames:
    :param common_win_step:
    :param signal: the input signal
    :param sample_rate: the sampling rate of the audio signal
    :param K: the fixed number of frames (rows of the feature vector)
    :param r: the distance between the windows relative to the window's size
    :param fixed: flag used to compute the fixed shape feature vector
    :return: the feature vector of the signal containing the static and dynamic features of the audio signal. The shape
    of the feature vector is fixed (K, 39) or variable (?,39). In the latter case we use a fixed window size, producing
    a variable number of frames depending on the duration (i.e. number of samples) of the input signal.
    """

    win_len, win_step = window_parameters(signal, sample_rate, common_win_step, fixed_number_of_frames, K, r, fixed)

    # Static features
    mfcc_feat = mfcc(signal=signal, samplerate=sample_rate, winlen=win_len, winstep=win_step, numcep=13,
                     nfilt=26, nfft=None, lowfreq=300, highfreq=None, preemph=0.95, ceplifter=0,
                     appendEnergy=True, winfunc=np.hamming)

    # Dynamic features
    d_mfcc_feat = delta(mfcc_feat, 2)
    dd_mfcc_feat = delta(d_mfcc_feat, 2)

    # Composition of the Feature vector
    feature_vector = np.concatenate((mfcc_feat, d_mfcc_feat, dd_mfcc_feat), axis=1)
    return feature_vector


def window_parameters(signal, sample_rate, common_win_len, common_win_step, K, r, fixed):
    """
    # Compute the size of the window for a particular signal in the dataset given the fixed number of frames.
    :param common_win_step:
    :param common_win_len:
    :param signal:
    :param sample_rate:
    :param K:
    :param r:
    :param fixed:
    :return:
    """

    if fixed:
        # if feature vector has a fixed shape (fixed number of frames), it is needed to compute the window's size
        Q = len(signal)
        L = np.ceil(Q / (1. + r * (K - 1.)))  # frame_len
        R = np.ceil(r * L)  # frame_step
        win_len = L / sample_rate
        win_step = R / sample_rate
    else:
        # if feature vector has not a fixed shape, a fixed size of the window is used, obtaining a different number of
        # frames for every signal.
        win_len = common_win_len
        win_step = common_win_step

    return win_len, win_step


def optimal_number_of_frames(dataset, sample_rate, win_len, win_step):
    """
    # Compute the best value of the number of frames given the shape of the dataset and an indication on what window
    # should me used for the application (audio signals' framing).
    :param dataset: set of audio signals
    :param sample_rate: sampling rate of the audio signals
    :param win_len: window size for the given application(audio signals' framing), indication of the wanted window size
    :param win_step: distance between windows
    :return:
    """

    r = win_step / win_len  # R/L, the fixed ratio between the window size and the step size
    # E.g. if L = 25 ms and R = 10 ms, then r = 0.4
    L_min = floor_to_power_of_2(sample_rate, win_len)
    Q_min = min_signal_length(dataset)
    K = np.floor(
        (1 / r) * (Q_min / (L_min + 1)) - (1 / r)) + 1  # number of frames, one of the dimensions of the feature vector
    return K, r


def min_signal_length(dataset):
    """
    Compute the minimum length of the signal in the dataset (compute the minimum value's length in the dictionary)
    :param dataset: set of audio signals
    :return: length of the shortest signal
    """

    min_length = min([len(n) for n in dataset.values()])
    return min_length


def floor_to_power_of_2(sample_rate, win_len):
    """
    Compute the greatest power of two smaller than the number of samples of the given window.
    :param sample_rate: sampling rate of the audio signals
    :param win_len: window size for the given application(audio signals' framing), indication of the wanted window size
    :return: greater power of 2 smaller than the number of samples of the given window
    """

    frame_len = sample_rate * win_len
    power = 1
    while power < frame_len:
        power *= 2
    return power / 2
