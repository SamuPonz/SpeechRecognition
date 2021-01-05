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
