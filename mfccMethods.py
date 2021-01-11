from _library_Mfccs import mfcc, delta, calculate_nfft
import numpy as np

# This method is placed here in order not to modify the library file "_library_Mfccs.py"


def min_signal_length(dataset):
    min_length = min([len(n) for n in dataset.values()])
    return min_length


def floor_to_power_of_2(sample_rate, win_len):
    frame_len = sample_rate*win_len
    pow = 1
    while pow < frame_len:
        pow *= 2
    return pow/2


def optimal_parameters(dataset, sample_rate, win_len, win_step):
    r = win_step/win_len  # R/L, the fixed ratio between the window size and the step size
    # E.g. if L = 25 ms and R = 10 ms, then r = 0.4
    L_min = floor_to_power_of_2(sample_rate, win_len)
    Q_min = min_signal_length(dataset)
    K = np.floor((1/r)*(Q_min/(L_min+1)) - (1/r)) + 1  # number of frames, one of the dimensions of the feature vector
    return K, r


def window_parameters(signal, sample_rate, K, r):
    Q = len(signal)
    L = np.ceil(Q/(1. + r*(K - 1.)))  # frame_len
    R = np.ceil(r*L)  # frame_step
    win_len = L/sample_rate
    win_step = R/sample_rate
    return win_len, win_step


def mfcc_processing(signal, sample_rate, K, r, fixed):
    if not fixed:
        win_len, win_step = window_parameters(signal, sample_rate, K, r)
    else:
        win_len = 0.025  # ms
        win_step = 0.01  # ms
    mfcc_feat = mfcc(signal=signal, samplerate=sample_rate, winlen=win_len, winstep=win_step, numcep=13,
                     nfilt=26, nfft=None, lowfreq=300, highfreq=None, preemph=0.95, ceplifter=0,
                     appendEnergy=True, winfunc=np.hamming)
    d_mfcc_feat = delta(mfcc_feat, 2)
    dd_mfcc_feat = delta(d_mfcc_feat, 2)
    feature_vector = np.concatenate((mfcc_feat, d_mfcc_feat, dd_mfcc_feat), axis=1)
    return feature_vector
