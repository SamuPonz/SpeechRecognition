import os
import time
from ast import literal_eval

import pandas as pd
from natsort import natsorted
from playsound import playsound
from scipy.io import wavfile
from sklearn.model_selection import train_test_split
import fileManagment as fm
import numpy as np


def load_dataset(dataset_directory, audiofiles_directory, dataset_name):
    # Loads the dataset or creates a new one using the audio files stored in the specified directory

    dataset_file = dataset_directory / dataset_name

    if not (dataset_file.is_file()):  # if file does not exist
        print("No saved dataset found. Building a new dataset named: ", dataset_name)
        dataset = read_audio_files(audiofiles_directory)
        fm.create_file(dataset_name, dataset)  # saves the dictionary in a file
    # return literal_eval(fm.readFile(dataset_name))  # Safely evaluate an expression node or a string containing a
    # Python literal or container display. The string or node provided may only consist of the following Python literal
    # structures: strings, bytes, numbers, tuples, lists, dicts, sets, booleans, None, bytes and sets.
    return fm.read_file(dataset_name)  # this is now a string, not a dictionary, nevertheless dict methods work fine


def read_audio_files(directory):
    # Reads the audio file in the specified directory
    # Returns a dictionary {filename : audio signal}

    audio_signals = dict()  # Dictionary of the audio signals {filename : audio signal}
    for subName in os.listdir(directory):
        print("reading the '" + subName + "' directory...")
        sub_directory = directory / subName
        # os.listdir(sub_directory).sort(key=lambda f: int(re.sub('\D', '', f)))  # sorting the files, not required
        print(sub_directory)  # debug
        sorted_files = natsorted(os.listdir(sub_directory))
        for filename in sorted_files:
            # print(filename)  # debug
            sample_rate, audio = wavfile.read(sub_directory / filename)
            audio_signals[filename] = audio  # add the file in the dictionary
    return audio_signals  # returns the dictionary


def load_noise(noise_directory, audiofiles_directory, noise_name):
    # Loads the dataset or creates a new one using the audio files stored in the specified directory

    noise_file = noise_directory / noise_name

    if not (noise_file.is_file()):  # if file does not exist
        print("No saved noise signals found. Building a new file named: ", noise_name)
        noise = read_noise_files(audiofiles_directory)
        fm.create_file(noise_name, noise)  # saves the dictionary in a file
        # return literal_eval(fm.readFile(dataset_name))  # Safely evaluate an expression node or a string containing a
        # Python literal or container display. The string or node provided may only consist of the following Python literal
        # structures: strings, bytes, numbers, tuples, lists, dicts, sets, booleans, None, bytes and sets.
    return fm.read_file(noise_name)  # this is now a string, not a dictionary, nevertheless dict methods work fine


def read_noise_files(directory):
    # Reads the audio file in the specified directory
    # Returns a dictionary {filename : audio signal}

    noise_signals = dict()  # Dictionary of the audio signals {filename : audio signal}
    for filename in os.listdir(directory):
        # print(filename)  # debug
        sample_rate, audio = wavfile.read(directory / filename)
        noise_signals[filename] = audio  # add the file in the dictionary
    return noise_signals  # returns the dictionary


# Function used to rename files in a directory:

# def reshapeDataset():
# directory = r'C:\Users\Samuele\Desktop\Workspace Tesi\Comandi Samu\ComandiDatasetRinominati'
# for subName in os.listdir(directory):
#     i = 1
#     print(subName)
#     subDirectory = directory + '\\' + subName
#     print(subDirectory)
#     for filename in os.listdir(subDirectory):
#         newName = subDirectory.split("ComandiDatasetRinominati\\", 1)[1] + str(i) + '.wav'
#         i = i + 1
#         print(filename)
#         print(newName)
#         os.rename(subDirectory + '\\' + filename, subDirectory + '\\' + newName)
#     return
