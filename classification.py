import pandas as pd
import fileManagment as fm
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import numpy as np


# The machine learning pipeline has the following steps: preparing data, creating training/testing sets, instantiating
# the classifier, training the classifier, making predictions, evaluating performance, tweaking parameters.

#check out k-fold cross validation


def classification_method(features, labels):

    # Data preparation
    dataset_matrix = data_preparation(features, labels)

    # Classification starts here:
    # (from SciKit docs: ) SVC, NuSVC and LinearSVC take as input two arrays: an array X of shape(n_samples, n_
    # features)
    # holding the training samples, and an array y of class labels(strings or integers), of shape (n_samples)

    X = dataset_matrix[:, :-1]
    Y = dataset_matrix[:, -1]

    # Creation of the training/testing sets
    x_train, x_val, x_test, y_train, y_val, y_test = data_split(X, Y)

    # Multi-class classification:
    # SVC and NuSVC implement the “one - versus - one” approach for multi-class classification. In total,
    # n_classes * (n_classes - 1) / 2 classifiers are constructed and each one trains data from two classes.
    # To provide a consistent interface with other classifiers, the decision_function_shape option allows to
    # monotonically transform the results of the “one-versus-one” classifiers to a “one-vs-rest” decision function of
    # shape (n_samples, n_classes).
    clf = SVC(decision_function_shape='ovo')
    clf.fit(x_train, y_train)

    dec = clf.decision_function([[1]])
    print(dec.shape[1])  # 10 classes: 10*(10-1)/2 = 45 classifiers

    clf.decision_function_shape = "ovr"
    dec = clf.decision_function([[1]])
    print(dec.shape[1])  # 10 classes

# ------------------------------------------------------------------


def data_preparation(features, labels):
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

    # convertion to data frame
    # dataframe = pd.DataFrame(dataset_matrix, columns=titles)
    # print(dataframe)

    return dataset_matrix


def data_split(dataX, dataY):
    train_ratio = 0.75
    validation_ratio = 0.15
    test_ratio = 0.10

    # train is now 75% of the entire data set
    # the _junk suffix means that we drop that variable completely
    x_train, x_test, y_train, y_test = train_test_split(dataX, dataY, test_size=1 - train_ratio)

    # test is now 10% of the initial data set
    # validation is now 15% of the initial data set
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test,
                                                    test_size=test_ratio / (test_ratio + validation_ratio))

    return x_train, x_val, x_test, y_train, y_val, y_test


def old_data_split(features, train_size=0.7):
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