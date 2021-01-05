from _library_Mfccs import mfcc, delta
import numpy as np

# This method is placed here in order not to modify the library file "_library_Mfccs.py"


def mfccProcessing(signal, sample_rate):
    mfcc_feat = mfcc(signal=signal, samplerate=sample_rate, winlen=0.025, winstep=0.01, numcep=13,
                     nfilt=26, nfft=512, lowfreq=300, highfreq=None, preemph=0.95, ceplifter=0,
                     appendEnergy=True, winfunc=np.hamming)
    d_mfcc_feat = delta(mfcc_feat, 2)
    dd_mfcc_feat = delta(d_mfcc_feat, 2)
    feature_vector = np.concatenate((mfcc_feat, d_mfcc_feat, dd_mfcc_feat), axis=1)
    return feature_vector
