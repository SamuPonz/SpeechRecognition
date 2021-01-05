import numpy as np

# This files contains all the methods regarding the pre-processing phase of the signals
# Here the dataset containing the raw signals is taken:
# at first all signals are filtered using a BP filter with a
# band [300,3400] Hz, BW = 3100 Hz. Check which implementation is better, more efficient. Up to now I can think about
# FIR filter realized with LCCDE, or with windowing design.
# Then the silence removal is implemented, have to understand more about this...
# Finally the end point is detected.


# Need for a function that acts on all the dictonary and dor functions that act on single signals
# Work on the single signal function needed for the test/recognition phase.
# The function working on the entire dataset is needed in the training phase, here it is set the noise threshold
# used in the test/recognition phase.

# Create a file with the data noise level!
# Maybe a file with all the information of the dataset will be used, maybe JSON.

# Normalized Dataset only needed for RTD and not MFCC??
import scipy


def preProcessing(dataset, sample_rate):

    # The Dataset is normalized: all the signals are rescaled in the range [-1,1] in order to apply the RTD.
    # Indeed this is one of the assumptions about the signal on which it applied the RTD.
    normalized_dataset = {k: newNormalization(v) for k, v in dataset.items()}

    # It is useful do to apply a filter that can be seen as an "economic" segmentation: in order to reduce the samples
    # of the signals all the samples whose absolute value is below a particular threshold are discarded. It is crucial
    # to select the threshold properly. Since in the references was not specified the method usd to set the threshold,
    # what it is done here is a personal solution. In this method, it is simply computed the mean on the first n
    # elements of every signal. The means are collected in a vector and the threshold is set to E[means] + C*std[means],
    # where C is a particular coverage factor which value is set manually, depending on the dataset.
    C = 0.5
    dataset_noise_level = silenceThreshold(normalized_dataset, C)

    segmented_dataset = {k: segmentation(v, dataset_noise_level) for k, v in normalized_dataset.items()}

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

    return segmented_dataset


def newNormalization(x):

    # Casting of the signal due to the fact that we are re-scaling it in a range [-1; 1], we cannot use integers.
    x = x.astype(np.single)

    # Normalization: Every value is divided by the max absolute value
    x = x / np.abs(x).max()

    # The following method is not correct, it modifies the distributions of values in the signals
    # x = np.where(x >= 0, x / np.max(x), -x / np.min(x))

    # The same method is written in another way
    # x_n = np.zeros(x.shape[0])
    # for i in range(x.shape[0]):
    #     if x[i] < 0:
    #         x_n[i] = -x[i]/np.min(x)
    #     elif x[i] > 0:
    #         x_n[i] = x[i]/np.max(x)
    #     else:
    #         break

    return x


def oldNormalization(x):
    # This method does not modify the signal distribution in [-1,1] but the baseline is different for every signal

    # normalization in [-1,1]: x_normalized = (x-min(x))/(max(x)-min(x))
    # the input array x is a int16 type
    # the output array scaled_x is a float32 type

    x = x.astype(np.single)  # casting is done to avoid overflow in the following operations
    num = x - int(np.min(x))
    den = int(np.max(x)) - int(np.min(x))
    scaled_x = np.multiply(np.divide(num, den), 2.0) - 1

    # print("expected num = x - x_min: ", x[111] - int(np.min(x)), "with x: ", x[111], "and x_min: ", int(np.min(x)))
    # print("actual num ", num[111])
    # print("expected scaled1: ", np.divide(x[111] - int(np.min(x)), den))
    # print("actual scaled1: ", np.divide(num, den))

    return scaled_x


def silenceThreshold(dataset, C):
    # --- Estimating the silence threshold taking the first 500 samples of each signal in the dataset ---
    # This may not be the best solution...

    # C = 0 # coverage factor

    means = []
    # For evaluating the silence level, the first 500 samples of the
    # signals in the dataset are taken and the mean of every of them is computed.
    # Now is possible to work with the population of the averages and take the silence
    # level as a threshold of the absolute value of the signals: this threshold represents
    # a particular confidence interval of 99.7% (k = 3).

    for signal in dataset.values():
        mean = np.mean(signal[:500])
        means.append(mean)
    noise = np.mean(means) + C * np.std(means)
    print("mean of the means: ", np.mean(means))
    print("standard deviation of the means: ", np.std(means))
    print("noise level estimated: ", noise)
    return noise


def segmentation(signal, noise):
    # The segmentation is performed deleting the silence time intervals in the audio signals. This reduces the length
    # of the time series, leaving only meaningful samples.
    # noise = the estimated silence level, set manually observing the dataset
    return signal[np.abs(signal) > noise]


# //////////////////////////////////////////////////////////////


def noiseReduction(signal, sample_rate):

    # Try this implementation, advised for low cost hardware
    # central_frequency = 1500  # Hz
    # samples =
    # Q = samples/(sample_rate/central_frequency)  # or defined some other way
    # w_c = 2 * np.pi * central_frequency / sample_rate
    # beta = np.cos(w_c)
    # BW = w_c / Q
    # alpha = (1. - np.sin(BW)) / np.cos(BW)
    # G = (1. - alpha) / 2.
    # filtered_signal = scipy.signal.lfilter([G, 0, -G], [1, -beta * (1 + alpha), alpha], signal)
    # return filtered_signal
    return


def silenceRemoval():
    return


def endPointDetection():
    return

