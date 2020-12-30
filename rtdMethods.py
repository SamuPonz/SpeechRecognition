import numpy as np


def rtdNew(signal, window_size, scaled):
    # Computation of the Reaction Diffusion Transform.
    # This method returns an approximation of the spectrum of the input signal
    # Parameters:
    # - signal: an input signal, representing an utterance of a vocal command
    # - w: the fixed number of samples in an output spectrum, 
    #      this parameter regulates the trade-off between time and frequency resolution
    # - flag: if '1' it is computed the scaled version of the transform, it is '0' by default.

    signal_samples = np.size(signal)  # number of samples of the signal
    n = int(np.log2(window_size))  # Casting to an int will truncate toward zero. floor() will truncate toward negative
    # infinite. Therefore there is no problem as long as the argument of the casting is positive (always in our case,
    # where window_size is always greater than 1!)
    windows = signal_samples // window_size  # number of windows of the signal (floor division)
    useful_samples = windows * window_size  # the total number of samples used to compute the spectrums
    # the other samples of the signal are discarded. In this way, the number of samples is a multiplier of w.

    matrix = np.reshape(signal[0:useful_samples], (windows, window_size))
    # the input signal is rearranged in a matrix
    # w is the number of samples of every window
    # the approximated spectrum of evey window is computed and stored as a column in a new matrix called 'spectrum'

    spectrogram = np.zeros((n - 3, windows))
    # in this matrix are stored all the spectrums computed
    # every spectrum has a size equal to n-3, log2(k)-3

    for i in range(0, windows):  # for every window of the signal
        values = matrix[i, :]  # take a window of the signal, a row of the windows matrix
        for k in range(0, n - 3):  # k is one of the bins of the approximated spectrums
            delay = 2 ** k  # delay considered to compute this particular clustering coefficient
            t = np.array(range(delay, window_size - delay - 1))  # array of time considered

            # The following command-line computes the absolute value of the Lagrange operator.
            # This is computed in a different way in respect to the classic RTD. This RTD_new uses all the samples
            # in the array t, the classic RTD the array t is down-sampled by the delay factor.
            diffusion = np.abs(values[t - delay] + values[t + delay] - 2 * values[t])

            # the final step in the computation of the k-th clustering coefficient of the spectrum of the i-th window
            if scaled == 0:  # RDT
                spectrogram[k, i] = np.mean(diffusion) / 4
            elif scaled == 1:  # SRDT
                # scaled version of the transform, where the Laplacian operator is substituted with its scaled version
                spectrogram[k, i] = np.mean(
                    diffusion / (np.abs(values[t - delay]) + np.abs(values[t + delay]) + 2 * np.abs(values[t]))) / 4
        # at the end of the inner for cycle, the spectrum of the i-th window has been computed (column vector)
    # at the end of the outer for cycle, the approximated spectrogram of the input signal is computed and returned
    return spectrogram


def buildFeatureVector(spectrogram, M, key):
    print(key)
    k = spectrogram.shape[0]  # number of channels of every spectrum in the spectrograms
    T = spectrogram.shape[1]  # number of spectrums (computed from the windows) in the spectrogram
    Ts = T // M  # number of spectrums per segment on which it is done the average, has to be at least 1
    # i.e. T must be greater than or equal to M.
    feature_vector = np.empty((k, M), dtype=np.single, order='C')  # since the final dimensions of the fv are fixed we
    # can use pre-allocation which is more efficient
    for j in np.arange(0, M):  # with np.arange j goes from 0 to M-1
        subspectral_sequence = spectrogram[:, (j * Ts):(j * Ts + Ts)]  # Ts spectrums are taken from the spectrogram
        average_spectrum = np.mean(subspectral_sequence, axis=1)  # the average spectrum of this sequence is computed
        feature_vector[:, j] = average_spectrum  # the spectrograms do not overlap
    return feature_vector


def newNormalization(x):
    # This method reduces the dynamic range of the signal... don't know if it is good...

    x = x.astype(np.single)
    x = np.where(x >= 0, x / np.max(x), -x / np.min(x))
    # x = x/np.abs(x).max()
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
    # This method retains the ynamic range of the signal in [-1,1] but the baseline is different for every signal

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
