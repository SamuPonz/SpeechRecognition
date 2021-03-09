import os
from tkinter import filedialog, Tk

from playsound import playsound
from scipy.io import wavfile
from preProcessing import noise_reduction, new_normalization, silence_removal
from rdtMethods import rdt_new, build_feature_vector
from util import measure
from tkinter.filedialog import askopenfilename


@measure
def recognition(clf, selected_commands, global_label_indices):
    # reads the file
    root = Tk()
    root.withdraw()
    root.wm_attributes("-topmost", 1)
    print("command classes that can be recognised are: ")
    print(selected_commands)
    # print("Please record yourself")
    # audio_file = record_audio()
    print("Please select a .wav audio file")
    audio_file = filedialog.askopenfilename(parent=root)  # shows dialog box and return the path
    playsound(audio_file)
    sample_rate, signal = wavfile.read(audio_file)
    audio_file = os.path.basename(audio_file)


    # applies pre-processing
    filtered = noise_reduction(signal, sample_rate)
    normalized = new_normalization(filtered)
    segmented = silence_removal(audio_file, normalized, sample_rate)

    # extracts features
    K = 4
    M = 10
    scaled = 1
    spectrogram = rdt_new(audio_file, segmented, 2 ** (K + 3), scaled)
    features = build_feature_vector(audio_file, spectrogram, M)

    # classification
    features = features.transpose()
    global_feature_vector = features.flatten()
    y_predicted = clf.predict(global_feature_vector.reshape(1, -1))
    command_class = next(key for key, value in global_label_indices.items() if value == y_predicted)
    print("The vocal command is: " + command_class.upper())

