import warnings

# import librosa
# import skimage.transform
from _library_paulstretch_mono import paulstretch
from scipy.signal import butter, sosfilt
import _library_Sigproc as sigproc
import numpy as np
import fileManagment as fm
from util import measure

# This file contains all the methods regarding the pre-processing phase of the signals
# Here the dataset containing the raw signals is taken:
# at first all signals are filtered using a BP filter with a
# band [300,3400] Hz, BW = 3100 Hz. Check which implementation is better, more efficient.
# Then the silence removal is implemented, have to understand more about this...

# Need for a function applied on all the dictionary and dor functions that act on single signals
# Work on the single signal function needed for the test/recognition phase.
# The function working on the entire dataset is needed in the training phase, here it is set the noise threshold
# used in the test/recognition phase.

# Create a file with the data noise level!
# Maybe a file with all the information of the dataset will be used, maybe JSON.

# Normalized Dataset only needed for RTD and not MFCC??

# ----------------------------------------- Loading the pre-processed data --------------------------------------------


@measure
def load_preprocessed_dataset(dataset, sample_rate, dataset_directory, dataset_name):
    fine_dataset_name = "fine_" + dataset_name
    fine_dataset_file = dataset_directory / fine_dataset_name

    if fine_dataset_file.is_file():
        return fm.read_file(fine_dataset_file)
    else:
        print("No saved pre-processed dataset found. New pre-processing...")
        fine_dataset = pre_processing(dataset, sample_rate)
        fm.create_file(fine_dataset_name, fine_dataset)
        return fine_dataset


@measure
def load_segmented_dataset(dataset, sample_rate, dataset_directory, dataset_name):
    segmented_dataset_name = "segmented_" + dataset_name
    segmented_dataset_file = dataset_directory / segmented_dataset_name

    if segmented_dataset_file.is_file():
        return fm.read_file(segmented_dataset_file)
    else:
        print("No saved segmented dataset found. New segmentation...")
        segmented_dataset = segmentation(dataset, sample_rate)
        fm.create_file(segmented_dataset_name, segmented_dataset)
        return segmented_dataset


# def load_stretched_dataset(dataset, dataset_directory, dataset_name):
# #     stretched_dataset_name = "stretched_" + dataset_name
# #     stretched_dataset_file = dataset_directory / stretched_dataset_name
# #
# #     if stretched_dataset_file.is_file():
# #         return fm.read_file(stretched_dataset_file)
# #     else:
# #         print("No saved stretched dataset found. New stretching on the data...")
# #         stretched_dataset = new_stretching_correction(dataset)
# #         fm.create_file(stretched_dataset_name, stretched_dataset)
# #         return stretched_dataset


# ------------------------------------- Performing pre-processing on the data -----------------------------------------


# 1st step
def pre_processing(dataset, sample_rate):
    filtered_dataset = {k: noise_reduction(v, sample_rate) for k, v in dataset.items()}
    normalized_dataset = {k: new_normalization(v) for k, v in filtered_dataset.items()}
    return normalized_dataset


# 2nd step
def segmentation(normalized_dataset, sample_rate):
    segmented_dataset = {k: silence_removal(k, v, sample_rate) for k, v in normalized_dataset.items()}
    return segmented_dataset


# 3rd step
# def new_stretching_correction(segmented_dataset):
#     # Not a simple interpolation, this correctly stretches the signals preserving the pitch!
#     stretched_dataset = {k: librosa_stretch(k, v) for k, v in segmented_dataset.items()}
#     return stretched_dataset


# ---------------------------------- Inner function working on single audio signals -----------------------------------

# def librosa_stretch(name, signal):
#     # Debug:
#     # print(name)
#     # print("n. of samples: %d" % signal.shape[0])
#
#     initial_length = signal.shape[0]
#     target_length = 12288
#     stretch_factor = initial_length/target_length
#     with warnings.catch_warnings():
#         warnings.filterwarnings(action="ignore", category=UserWarning)
#         stretched_signal = librosa.effects.time_stretch(signal, stretch_factor)
#     return stretched_signal


def noise_reduction(signal, sample_rate):
    lowcut = 300  # Hz
    highcut = 4000  # Hz
    order = 10
    filtered_signal = butter_bandpass_filter(signal, lowcut, highcut, sample_rate, order)
    return filtered_signal


def new_normalization(x):
    # Casting of the signal due to the fact that we are re-scaling it in a range [-1; 1], we cannot use integers.
    x = x.astype(np.single)

    # Normalization: Every value is divided by the max absolute value
    x = x / np.abs(x).max()
    return x


def silence_removal(name, signal, sample_rate, win_func=lambda x: np.ones((x,))):

    # Debugging (If you use it, remember to add the name argument!)
    # if name == "on27":
    #     print()

    # Define thresholds for energy and zero crossing rate
    energy_th = 0.2  # if higher, more segmentation
    zero_crossing_th = 65  # if lower, more segmentation
    # Define window dimension in time for framing the signal, computing the number of samples
    win_len = 0.01  # 10 ms
    win_step = 0.01  # 10 ms # if equal to win_len, no overlap
    # windows of 25 ms without overlapping generate frames of 400 samples, up to 40 frames in this dataset
    frame_len = win_len * sample_rate
    frame_step = win_step * sample_rate

    # Signal framing
    frames = sigproc.framesig(signal, frame_len, frame_step, win_func)

    # Energies of the frames
    energies = np.sum(np.square(frames), 1)  # this stores the total energy in each frame (each row)

    # Zero crossing rates of the frames
    zero_crosses = np.zeros(frames.shape[0])
    for i in np.arange(0, frames.shape[0]):
        zero_crosses[i] = (np.diff(np.sign(frames[i, :])) != 0).sum()

    # Concatenate the parameters to the relative frames in order to easy delete frames
    energy_T = np.transpose([energies])
    zcr_T = np.transpose([zero_crosses])
    frames = np.concatenate((energy_T, zcr_T, frames), axis=1)

    # Delete frames which have low energy and high zero crossing rate
    frames = frames[np.logical_and(frames[:, 0] > energy_th, frames[:, 1] < zero_crossing_th)]
    # Conditional statements like data[:,0] < 25 create boolean arrays that track, element-by-element, where the
    # condition in an array is true or false. Index numpy arrays with these boolean arrays. This kind of conditional
    # indexing allows to extract the rows (or columns, or elements).
    # Logical tools to combine boolean arrays element-by-element. Regular and, or, and not statements don't work because
    # they try to combine the boolean arrays together as a whole. Numpy provides a set of these tools for use in the
    # form of np.logical_and, np.logical_or, and np.logical_not. It is possible to combine boolean arrays element-wise
    # to find rows that satisfy more complicated conditions.

    # Delete parameters (energies and zero crossing rate) stored in the matrix of frames
    frames = frames[:, 2:]
    # Reconstruct the signal
    segmented_signal = sigproc.deframesig(frames, frames.size, frame_len, frame_step, win_func)

    # If the original signal had not a number of samples equal to a multiple of frame_len (in case of no overlapping)
    # then the "frames" matrix will be padded with zeros in the last row. In the case in which the last row is not
    # elided by the segmentation stage, this zeros cause problems in the SRDT (S stands for "scaled") computation,
    # therefore here we remove the zero-padding (which does not contain information).
    segmented_signal = np.trim_zeros(segmented_signal)
    segmented_signal = segmented_signal.astype(np.single)
    return segmented_signal


def butter_bandpass_filter(data, lowcut, highcut, sample_rate, order):
    nyq = 0.5 * sample_rate  # 8000 Hz
    low = lowcut / nyq  # 300/8000 = 0.0375
    high = highcut / nyq  # 4000/8000 = 0.5

    sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    y = sosfilt(sos, data)
    # filtfilt removes phase delay but it is slower
    # y = sosfiltfilt(sos, data)
    return y


# NOT USED --------------------------------------- Old solutions ------------------------------------------------------


def extreme_stretching_correction(segmented_dataset, sample_rate, stretch_factor, windowsize_seconds):
    # Not a simple interpolation, this correctly stretches the signals preserving the pitch!
    stretched_dataset = {k: extreme_stretch(k, v, sample_rate, stretch_factor, windowsize_seconds)
                         for k, v in segmented_dataset.items()}
    return stretched_dataset


def extreme_stretch(name, signal, sample_rate, stretch_factor, stretch_window_size):
    # Adapted version of the original version of this:

    ################################################################
    # Paul's Extreme Sound Stretch (Paulstretch) - Python version
    #
    # by Nasca Octavian PAUL, Targu Mures, Romania
    # http://www.paulnasca.com/
    #
    # http://hypermammut.sourceforge.net/paulstretch/
    ################################################################

    # print(name)
    # print("n. of samples: %d" % signal.shape[0])
    if signal.shape[0] < 4096:
        signal = paulstretch(sample_rate, signal, stretch_factor, stretch_window_size)
    return signal


# def old_stretching_correction(segmented_dataset, min_length):
#     stretched_dataset = {k: interpolation_stretch(k, v, min_length) for k, v in segmented_dataset.items()}
#     return stretched_dataset


# def interpolation_stretch(name, signal, min_length):
#     # This is a simple interpolation of the signals, used to increment the number of samples of particularly short
#     # signals and increase performance.
#     # This operation does not preserve the frequency content of the signals (it changes the pitch) and sadly it does
#     # not increase the performance of the algorithm. It was to good to be true!
#
#     # print(name)
#     # print("n. of samples: %d" % signal.shape[0])
#
#     if signal.shape[0] < min_length:
#         # plt.figure()
#         # plt.plot(signal)
#         signal = skimage.transform.resize(signal, (min_length, 1), anti_aliasing=False).squeeze()
#         # plt.figure()
#         # plt.plot(signal)
#     return signal


def old_pre_processing(dataset):

    # The Dataset is normalized: all the signals are rescaled in the range [-1,1] in order to apply the RTD.
    # Indeed this is one of the assumptions about the signal on which it applied the RTD.
    normalized_dataset = {k: new_normalization(v) for k, v in dataset.items()}

    # It is useful do to apply a filter that can be seen as an "economic" segmentation: in order to reduce the samples
    # of the signals all the samples whose absolute value is below a particular threshold are discarded. It is crucial
    # to select the threshold properly. Since in the references was not specified the method usd to set the threshold,
    # what it is done here is a personal solution. In this method, it is simply computed the mean on the first n
    # elements of every signal. The means are collected in a vector and the threshold is set to E[means] + C*std[means],
    # where C is a particular coverage factor which value is set manually, depending on the dataset.
    C = 3
    dataset_noise_level = old_silence_threshold(normalized_dataset, C)

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


def old_normalization(x):
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


def old_silence_threshold(dataset, C):
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


def old_segmentation(signal, noise):
    # The segmentation is performed deleting the silence time intervals in the audio signals. This reduces the length
    # of the time series, leaving only meaningful samples.
    # noise = the estimated silence level, set manually observing the dataset
    return signal[np.abs(signal) > noise]


# Alternative method to implement the band-pass filter
# def simple_band_pass(signal,  lowcut, highcut, sample_rate):
#     # Second order band pass filter
#     bandwidth = highcut - lowcut
#     central_frequency = bandwidth/2.
#     w_c = 2 * np.pi * central_frequency
#     bw = 2 * np.pi * bandwidth
#     beta = np.cos(w_c)
#     alpha = (1. - np.sin(bw)) / np.cos(bw)
#     G = (1. - alpha) / 2.
#     filtered_signal = scipy.signal.lfilter([G, 0, -G], [1, -beta * (1 + alpha), alpha], signal)
#     return filtered_signal

