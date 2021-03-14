import os
from tkinter import filedialog, Tk

from playsound import playsound
from scipy.io import wavfile
from preProcessing import noise_reduction, new_normalization, silence_removal
from rdtMethods import rdt_new, build_feature_vector
from util import measure
import joblib


def recognition(command_recordings_dir):
    # loads the model
    clf, selected_commands, global_label_indices = joblib.load('model.sav')

    # reads the new audio file to recognize
    root = Tk()
    root.withdraw()
    root.wm_attributes("-topmost", 1)
    print("command classes that can be recognised are: ")
    print(selected_commands)
    # print("Please record yourself")
    # audio_file = record_audio()
    print("Please select a .wav audio file")
    audio_file = filedialog.askopenfilename(parent=root, initialdir=command_recordings_dir)  # shows dialog box and return the path
    playsound(audio_file)
    algorithm(audio_file, clf, global_label_indices)


@measure
def algorithm(audio_filename, clf, global_label_indices):
    # reads audio file
    sample_rate, signal = wavfile.read(audio_filename)
    audio_filename = os.path.basename(audio_filename)
    segmented = preproc(signal,sample_rate,audio_filename)
    features = featextr(audio_filename, segmented)
    classif(features, clf, global_label_indices)


# functions are created in order to measure time execution

@measure
def preproc(signal, sample_rate, audio_filename):
    # applies pre-processing
    filtered = noise_reduction(signal, sample_rate)
    normalized = new_normalization(filtered)
    segmented = silence_removal(audio_filename, normalized, sample_rate)
    return segmented


@measure
def featextr(audio_filename, segmented):
    # extracts features
    K = 4
    M = 10
    scaled = 1
    spectrogram = rdt_new(audio_filename, segmented, 2 ** (K + 3), scaled)
    features = build_feature_vector(audio_filename, spectrogram, M)
    return features


@measure
def classif(features, clf, global_label_indices):
    # classification
    features = features.transpose()
    global_feature_vector = features.flatten()
    y_predicted = clf.predict(global_feature_vector.reshape(1, -1))
    command_class = next(key for key, value in global_label_indices.items() if value == y_predicted)
    print("The vocal command is: " + command_class.upper())
