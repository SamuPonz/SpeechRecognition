import os
from tkinter import filedialog, Tk

from playsound import playsound
from scipy.io import wavfile
from preProcessing import noise_reduction, new_normalization, silence_removal
from rdtMethods import rdt_new, build_feature_vector
from util import measure
import joblib
import wave
import pyaudio  # only works on the Raspberry


def recognise_after_record():
    clf, selected_commands, global_label_indices = joblib.load('model.sav')
    print("command classes that can be recognised are: ")
    print(selected_commands)
    input("Press a key to start the recognition: ")
    audio_file = record_audio()
    print("recording ended after 1 s")
    playsound(audio_file)
    algorithm(audio_file, clf, global_label_indices)


def recognise_audiofile(command_recordings_dir):
    # loads the model
    clf, selected_commands, global_label_indices = joblib.load('model.sav')
    print("command classes that can be recognised are: ")
    print(selected_commands)
    audio_file = open_file(command_recordings_dir, selected_commands)
    playsound(audio_file)
    algorithm(audio_file, clf, global_label_indices)


def record_audio():
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    RECORD_SECONDS = 1
    WAVE_OUTPUT_FILENAME = "output.wav"

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("* recording")

    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("* done recording")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    return WAVE_OUTPUT_FILENAME


def open_file(command_recordings_dir, selected_commands):
    # reads the new audio file to recognize
    root = Tk()
    root.withdraw()
    root.wm_attributes("-topmost", 1)
    # print("Please record yourself")
    # audio_file = record_audio()
    print("Please select a .wav audio file")
    audio_file = filedialog.askopenfilename(parent=root,
                                            initialdir=command_recordings_dir)  # shows dialog box and return the path
    return audio_file


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
