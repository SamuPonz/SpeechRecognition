import os
import time
import wave
from functools import wraps
from random import randrange

from playsound import playsound
from scipy.io import wavfile
import numpy as np
import rdtMethods as rtd
import matplotlib.pyplot as plt
# import pyaudio

import wave


def measure(func):
    # Decorator used to measure time execution of functions

    @wraps(func)
    def _time_it(*args, **kwargs):
        start = int(round(time.time() * 1000))
        try:
            return func(*args, **kwargs)
        finally:
            end_ = int(round(time.time() * 1000)) - start
            print(f"Total execution time of the method", func.__name__, "is : ", {end_ if end_ > 0 else 0}, " ms")
    return _time_it


def handle_warning(message, category, filename, lineno, file=None, line=None):
    # This function is used to interrupt the program execution when a warning occurs

    print('A warning occurred:')
    print(message)
    print('Do you wish to continue?')

    while True:
        response = input('y/n: ').lower()
        if response not in {'y', 'n'}:
            print('Not understood.')
        else:
            break

    if response == 'n':
        raise category(message)


def plottoez(name, dataset):
    signal = dataset.get(name)

    new_name = 'extracted_' + name + ".wav"
    wavfile.write(new_name, 16000, signal)
    playsound(new_name)

    fig, axs = plt.subplots(1, 1, constrained_layout=True)
    title_size = 10
    x_size = 10
    y_size = 10

    plt.plot(signal.transpose())
    plt.grid(True)
    plt.title('Raw signal', fontsize=title_size)
    plt.xlabel('samples', fontsize=x_size)
    plt.ylabel('amplitude', fontsize=y_size)


def plottoez2(name, segmented, rdt1, rdt2, rdt3, rdt4):

    fig, axs = plt.subplots(4, 1, constrained_layout=True)
    title_size = 10
    x_size = 10
    y_size = 10

    # plt.imshow(rdt3)
    # plt.grid(True)
    # plt.title('RDT k = 6', fontsize=title_size)
    # plt.xlabel('signal\'s segments', fontsize=x_size)
    # plt.ylabel('order', fontsize=y_size)

    axs[0].plot(segmented.transpose())
    axs[0].grid(True)
    axs[0].set_title('Segmented signal', fontsize=title_size)
    axs[0].set_xlabel('samples', fontsize=x_size)
    axs[0].set_ylabel('amplitude', fontsize=y_size)

    axs[1].imshow(rdt1)
    # axs[1].grid(True)
    axs[1].set_title('RDT k = 5', fontsize=title_size)
    axs[1].set_xlabel('signal\'s segments', fontsize=x_size)
    axs[1].set_ylabel('order', fontsize=y_size)

    axs[2].imshow(rdt2)
    # axs[2].grid(True)
    axs[2].set_title('RDT k = 6', fontsize=title_size)
    axs[2].set_xlabel('signal\'s segments', fontsize=x_size)
    axs[2].set_ylabel('order', fontsize=y_size)

    axs[3].imshow(rdt3)
    # axs[3].grid(True)
    axs[3].set_title('RDT k = 7', fontsize=title_size)
    axs[3].set_xlabel('signal\'s segments', fontsize=x_size)
    axs[3].set_ylabel('order', fontsize=y_size)


def parameter_examples():
    k = [1, 2, 3, 4, 5, 6, 7]
    k = np.array(k)
    M = 1280//(2**(k+3))

    # title_size = 10
    # x_size = 10
    # y_size = 10
    # plt.bar(k, M, width=0.3)
    # plt.grid(True)
    # plt.title('Possible values of M as a function of k', fontsize=title_size)
    # plt.xlabel('k', fontsize=x_size)
    # plt.ylabel('M', fontsize=y_size)

    title_size = 13
    x_size = 12
    y_size = 12
    x = []
    y = []

    for k in range(1, 8):
        t = 2**(k+3)
        for w in range(t, 1281):
            x.append(k)
            y.append(int(np.floor(1280/w)))
    plt.scatter(x, y)
    x = []
    y = []
    for k in range(1, 8):

        t = 2**(k+3)
        x.append(k)
        y.append(int(np.floor(1280/t)))

    plt.scatter(x, y, color='red')
    # plt.hlines(1.0, 0.0, 8.0, colors='red', linestyles='dotted', label='minimum')
    # plt.grid()
    plt.title('Possible combinations of k and M', fontsize=title_size)
    plt.xlabel('Possible values of k', fontsize=x_size)
    plt.ylabel('Possible values of M', fontsize=y_size)
    for xy in zip(x, y):  # <--
        plt.annotate('(%s, %s)' % xy, xy=xy, textcoords='data', size=13)  # <--

    plt.show()


def view_example(name, dataset, segmented_dataset, mfcc_features, rtd_features):
    # Look at a particular signal, selected by its name, adn its feature vectors

    signal = dataset.get(name)
    segmented_signal = segmented_dataset.get(name)
    mfcc_feature_vector = mfcc_features.get(name)
    rtd_feature_vector = rtd_features.get(name)

    new_name = 'extracted_' + name + ".wav"
    wavfile.write(new_name, 16000, signal)
    playsound(new_name)

    fig, axs = plt.subplots(3, 1, constrained_layout=True)
    # fig.suptitle('Example of an audio signal', fontsize=10)
    title_size = 10
    x_size = 10
    y_size = 10

    # plt.plot(signal.transpose())
    # plt.grid(True)
    # plt.title('Raw signal', fontsize=title_size)
    # plt.xlabel('samples', fontsize=x_size)
    # plt.ylabel('amplitude', fontsize=y_size)

    # axs[0].plot(signal.transpose())
    # axs[0].grid(True)
    # axs[0].set_title('Raw signal', fontsize=title_size)
    # axs[0].set_xlabel('samples', fontsize=x_size)
    # axs[0].set_ylabel('amplitude', fontsize=y_size)

    axs[0].plot(segmented_signal.transpose())
    axs[0].grid(True)
    axs[0].set_title('Segmented signal', fontsize=title_size)
    axs[0].set_xlabel('samples', fontsize=x_size)
    axs[0].set_ylabel('amplitude', fontsize=y_size)

    axs[1].imshow(mfcc_feature_vector)
    # axs[0].grid(True)
    axs[1].set_title('Classic RTD', fontsize=title_size)
    axs[1].set_xlabel('signal\'s segments', fontsize=x_size)
    axs[1].set_ylabel('order', fontsize=y_size)

    axs[2].imshow(rtd_feature_vector)
    # axs[1].grid(True)
    axs[2].set_title('Scaled RTD', fontsize=title_size)
    axs[2].set_xlabel('signal\'s segments', fontsize=x_size)
    axs[2].set_ylabel('order', fontsize=y_size)


def view_mfcc_example(name, dataset, mfcc_features):
    # Look at a particular signal, selected by its name, adn its feature vectors

    signal = dataset.get(name)
    mfcc_feature_vector = mfcc_features.get(name)

    new_name = 'extracted_' + name
    wavfile.write(new_name, 16000, signal)
    playsound(new_name)

    fig, axs = plt.subplots(3, 1, constrained_layout=True)
    fig.suptitle('Example of a signal and its mfcc feature vector', fontsize=10)
    title_size = 10
    x_size = 8
    y_size = 8

    axs[0].plot(signal.transpose())
    axs[0].grid(True)
    axs[0].set_title('Raw signal', fontsize=title_size)
    axs[0].set_xlabel('samples', fontsize=x_size)
    axs[0].set_ylabel('amplitude', fontsize=y_size)

    axs[1].imshow(mfcc_feature_vector.transpose())
    axs[1].set_title('mfcc feature vector', fontsize=title_size)
    axs[1].set_xlabel('time', fontsize=x_size)
    axs[1].set_ylabel('frequency', fontsize=y_size)

    axs[2].plot(mfcc_feature_vector.transpose())
    axs[2].set_title('mfcc feature vector', fontsize=title_size)
    axs[2].set_xlabel('time', fontsize=x_size)
    axs[2].set_ylabel('frequency', fontsize=y_size)



# def viewParticularRtdExample(name, normalized_dataset, segmented_dataset, window_size, M, scaled):
#     # Look at a particular signal, selected by its name, and all the steps of the Rtd feature extraction
#
#     raw_bad = normalized_dataset.get(name)
#     bad_signal = segmented_dataset.get(name)
#     spectrogram = rtd.rtdNew(bad_signal, window_size, scaled)
#     feature_vector = rtd.buildFeatureVector(spectrogram, M, name)
#     plt.figure(figsize=(15, 4))
#     plt.plot(raw_bad.transpose())
#     plt.figure(figsize=(15, 4))
#     plt.plot(bad_signal.transpose())
#     plt.figure(figsize=(15, 4))
#     plt.imshow(spectrogram)
#     plt.figure(figsize=(15, 4))
#     plt.imshow(feature_vector)


def view_random_rtd_example(j, dataset, normalized_dataset, segmented_dataset, dataset_spectrograms, dataset_features):
    # --- Observe an example of the rtd approach ---

    example0 = list(dataset.values())[j]
    example1 = list(normalized_dataset.values())[j]
    example2 = list(segmented_dataset.values())[j]
    # normalized = rtd.new_normalization(example3)
    example3 = list(dataset_spectrograms.values())[j]
    example4 = list(dataset_features.values())[j]

    wavfile.write('random_rtd_example.wav', 16000, example1)
    playsound('random_rtd_example.wav')

    # plotting section
    fig, axs = plt.subplots(5, 1, constrained_layout=True)
    fig.suptitle('Example of feature vector computed with RTD', fontsize=10)
    title_size = 10
    x_size = 8
    y_size = 8

    axs[0].plot(example0.transpose())
    axs[0].grid(True)
    axs[0].set_title('Raw signal', fontsize=title_size)
    axs[0].set_xlabel('samples', fontsize=x_size)
    axs[0].set_ylabel('amplitude', fontsize=y_size)

    axs[1].plot(example1.transpose())
    axs[1].grid(True)
    axs[1].set_title('Normalized signal', fontsize=title_size)
    axs[1].set_xlabel('samples', fontsize=x_size)
    axs[1].set_ylabel('amplitude', fontsize=y_size)

    axs[2].plot(example2.transpose())
    axs[2].grid(True)
    axs[2].set_title('Segmented signal', fontsize=title_size)
    axs[2].set_xlabel('samples', fontsize=x_size)
    axs[2].set_ylabel('amplitude', fontsize=y_size)

    # axs[3].plot(normalized.transpose())
    axs[3].imshow(example3)
    axs[3].set_title('Approximated spectrogram', fontsize=title_size)
    axs[3].set_xlabel('T time windows', fontsize=x_size)
    axs[3].set_ylabel('K cluster\ncoefficients', fontsize=y_size)

    axs[4].imshow(example4)
    axs[4].set_title('Feature vector', fontsize=title_size)
    axs[4].set_xlabel('M segments', fontsize=x_size)
    axs[4].set_ylabel('K cluster\ncoefficients', fontsize=y_size)


def view_random_mfcc_example(j, dataset, dataset_features):
    # --- Observe an example of the rtd approach ---

    example0 = list(dataset.values())[j]
    example1 = list(dataset_features.values())[j]

    wavfile.write('random_mfcc_example.wav', 16000, example0)
    playsound('random_mfcc_example.wav')

    # plotting section
    fig, axs = plt.subplots(3, 1, constrained_layout=True)
    fig.suptitle('Example of feature vector computed with MFCC', fontsize=10)
    title_size = 10
    x_size = 8
    y_size = 8

    axs[0].plot(example0.transpose())
    axs[0].grid(True)
    axs[0].set_title('Raw signal', fontsize=title_size)
    axs[0].set_xlabel('samples', fontsize=x_size)
    axs[0].set_ylabel('amplitude', fontsize=y_size)

    axs[1].plot(example1.transpose())
    axs[1].set_title('Feature vector', fontsize=title_size)
    axs[1].set_xlabel('static and dynamic Mel frequency cepstral coefficients', fontsize=x_size)
    axs[1].set_ylabel('time windows', fontsize=y_size)

    axs[2].imshow(example1.transpose())
    axs[2].set_title('Feature vector', fontsize=title_size)
    axs[2].set_xlabel('time windows', fontsize=x_size)
    axs[2].set_ylabel('static and\ndynamic coefficients', fontsize=y_size)


# def getVectorsParameters(dictio):
#     # Evaluation of the shapes of the signals: check what is the minimum size of the number of samples
#     sizes = []
#     for i in dictio.values():
#         T = i.shape[0]  # number of samples
#         sizes.append(T)
#     sizes = np.array(sizes, dtype=np.int)
#
#     print("The number of elements of the vectors ranges in: [", sizes.min(), ";", sizes.max(), "]")


def get_parameters(dictio):
    # Evaluation of the shapes of the data
    # E.g. checking what is the minimum size of the number of columns of the rtd spectrograms and select M in a smart
    # way to be sure that it is not greater than any column's size.

    rows_size = []
    columns_size = []
    # counter = 0
    for i in dictio.values():
        R = i.shape[0]  # number of rows (e.g. number of time windows in the spectrogram)
        # print(counter, " - ", R)
        rows_size.append(R)
        # counter += 1
        if len(i.shape) > 1:
            C = i.shape[1]  # number of columns (e.g. number of spectrums  in the spectrogram computed from the windows)
            columns_size.append(C)
    rows_size = np.array(rows_size, dtype=np.int)
    if len(i.shape) > 1:
        columns_size = np.array(columns_size, dtype=np.int)
    print("The number of rows ranges in: [", rows_size.min(), ";", rows_size.max(), "]")
    if len(i.shape) > 1:
        print("The number of columns ranges in: [", columns_size.min(), ";", columns_size.max(), "]")


def plot_dataset(dictio):
    # --- Plotting the entire dataset ---
    for i in range(10):
        i = 100*i
        counter = 1
        plt.figure(figsize=(15, 4))
        for key in list(dictio)[i:i + 99]:
            value = dictio.get(key)
            if counter == i+100:
                break
            plt.subplot(10, 10, counter)
            if len(value.shape) > 1:  # if data has a second dimension
                plt.imshow(value)
            else:
                plt.plot(value.transpose())
            plt.grid(True)
            plt.xlabel(key)

            counter += 1


def plot_noise(dictio):
    # --- Plotting the entire dataset ---
    plt.figure(figsize=(15, 4))
    counter = 1
    for value in list(dictio.values()):
        plt.subplot(10, 10, counter)
        plt.plot(value.transpose())
        plt.grid(True)
        counter += 1


def plot_class(command_class, dictio):
    # --- Plot an entire class of the dataset ---
    # command_class = 8  # select i from 0 to 9
    # if command_class != 0:
    #     command_index = command_class * 100
    # else:
    #     command_index = 0
    # counter = 0


    figure = plt.figure(figsize=(20, 10))
    figure.canvas.manager.full_screen_toggle()
    figure.suptitle(command_class+" commands", fontsize=10)
    foodict = {k: v for k, v in dictio.items() if k.startswith(command_class)}
    counter = 0
    for value in foodict.values():
        counter += 1
        # plt.grid(True)
        plt.subplot(10, (len(foodict) + 1) // 10, counter)
        if len(value.shape) > 1:  # if data has a second dimension
            plt.imshow(value)
        else:
            plt.plot(value.transpose())


def reproduce_raw_audio(name, dataset):
    example = dataset[name]
    wavfile.write('extracted.wav', 16000, example)
    playsound('extracted.wav')


def reproduce_normalized_audio(name, dataset, sample_rate):
    # example = dataset[name]
    # wavfile.write('extracted_'+name+'.wav', 16000, np.int16(example*16000).tostring())
    # playsound('extracted.wav')

    example = dataset[name]
    outfile = wave.open("extracted.wav", "wb")
    outfile.setsampwidth(2)
    outfile.setframerate(sample_rate)
    outfile.setnchannels(1)
    outfile.writeframes(np.int16(example*16000).tostring())
    outfile.close()

    playsound('extracted.wav')


def casting_influence(dataset):
    # --- Observe how casting influences the reproduction of audio file

    # The wavfile.read function used to create the dataset returns a nparray which type is the one which minimal
    # size needed.
    # We cast the numpy.int16 type to numpy.single to overcome overflow in the normalization procedure
    # After the casting, an audio signal cannot be reproduced anymore, why?
    # We maintain the type numpy.single because after normalization the signal varies in [-1,1], this cannot be
    # reconverted to any int type of any size

    j = 20  # select i in the range [0:999] # Consider signals 38 and 39.
    audio = list(dataset.values())[j]
    wavfile.write('audio.wav', 16000, audio)
    playsound('audio.wav')
    # time.sleep(1.2)
    casted_audio = audio.astype(np.single)
    wavfile.write('casted_audio.wav', 16000, casted_audio)
    playsound('casted_audio.wav')
    recasted_audio = casted_audio.astype(np.int16)
    wavfile.write('recasted_audio.wav', 16000, recasted_audio)
    playsound('recasted_audio.wav')

