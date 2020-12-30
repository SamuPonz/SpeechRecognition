import os
import time
from ast import literal_eval

import pandas as pd
from playsound import playsound
from scipy.io import wavfile
from sklearn.model_selection import train_test_split
import fileManagment as fm
import numpy as np


def loadDataset(dataset_directory, audiofiles_directory, dataset_name):
    # Loads the dataset or creates a new one using the audio files stored in the specified directory

    dataset_file = dataset_directory / dataset_name
    if not dataset_file.is_file():  # if file does not exist
        print("No saved dataset found. Building a new dataset named: ", dataset_name)
        dataset = readAudioFiles(audiofiles_directory)
        fm.createFile(dataset_name, dataset)  # saves the dictionary in a file
    # return literal_eval(fm.readFile(dataset_name))  # Safely evaluate an expression node or a string containing a
    # Python literal or container display. The string or node provided may only consist of the following Python literal
    # structures: strings, bytes, numbers, tuples, lists, dicts, sets, booleans, None, bytes and sets.
    return fm.readFile(dataset_name)  # this is now a string, not a dictionary, nevertheless dict methods work fine


def readAudioFiles(directory):
    # Reads the audio file in the specified directory
    # Returns a dictionary {filename : audio signal}

    audio_signals = dict()  # Dictionary of the audio signals {filename : audio signal}
    for subName in os.listdir(directory):
        print("reading the '" + subName + "' directory...")
        subDirectory = directory / subName
        # os.listdir(subDirectory).sort(key=lambda f: int(re.sub('\D', '', f)))  # sorting the files, not required
        # print(subDirectory)  # debug
        for filename in os.listdir(subDirectory):
            # print(filename)  # debug
            sample_rate, audio = wavfile.read(subDirectory / filename)
            audio_signals[filename] = audio  # add the file in the dictionary
    return audio_signals  # returns the dictionary


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
