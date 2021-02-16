import pandas as pd
import fileManagment as fm
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import numpy as np


# The machine learning pipeline has the following steps: preparing data, creating training/testing sets, instantiating
# the classifier, training the classifier, making predictions, evaluating performance, tweaking parameters.

#check out k-fold cross validation


def classification_method(labels, features):
    label_indices = {}
    i = 0
    for item in labels:
        if i > 0 and item in label_indices:
            continue
        else:
            i = i + 1
            label_indices[item] = i

    n_of_features = next(iter(features.values())).shape[1]
    n_of_segments = 0
    for value in features.values():
        n_of_segments += value.shape[0]

    # convert to matrix: pre-allocation used
    dataset_matrix = np.zeros(shape=(n_of_segments, n_of_features + 1))
    last_filled = 0
    for key in features:
        command_class = ''.join([i for i in key if not i.isdigit()])
        command_index = label_indices[command_class]
        value = features[key]
        dataset_matrix[last_filled:last_filled+value.shape[0], 0:-1] = value
        dataset_matrix[last_filled:last_filled+value.shape[0], -1] = command_index
        # This works only in the case of fixed n_of_segments:
        # t = 0
        # dataset_matrix[value.shape[0]*t:value.shape[0]*(t+1), 0:-1] = value
        # dataset_matrix[value.shape[0]*t:value.shape[0]*(t+1), -1] = command_index
        # t += 1
        last_filled += value.shape[0]

    # add a first row with titles or convert into a Pandas dataframe
    titles = ["feature"+str(i+1) for i in range(next(iter(features.values())).shape[1])]
    titles.append("label")
    print(titles)
    dataset_matrix_with_titles = np.vstack([titles, dataset_matrix])

    # convert to data frame (copied, to be revised)
    dataframe = pd.DataFrame(dataset_matrix, columns=titles)
    print(dataframe)

# ------------------------------------------------------------------


# Old functions, here dictionaries are used.

def data_split(features, train_size=0.7):
    # Creating testing and training sets

    s = pd.Series(features)
    training_dataset, test_dataset = [i.to_dict() for i in train_test_split(s, train_size=train_size)]
    return training_dataset, test_dataset


def load_subsets(features, features_directory):
    # Loads saved train and test sets, creates them if there are none
    train_name = "train_features.p"
    test_name = "test_features.p"
    train_file = features_directory / train_name
    test_file = features_directory / test_name
    if not train_file.is_file() or not test_file.is_file():
        print("No saved train/test sets found. Building two new random sets...")
        train_dataset, test_dataset = data_split(features, train_size=0.7)
        fm.create_file(train_name, train_dataset)
        fm.create_file(test_name, test_dataset)
    return fm.read_file(train_file), fm.read_file(test_file)