import warnings

from pandas._libs import json
from sklearn.exceptions import UndefinedMetricWarning

import pandas as pd
import fileManagment as fm
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt


# The machine learning pipeline has the following steps: preparing data, creating training/testing sets, instantiating
# the classifier, training the classifier, making predictions, evaluating performance, tweaking parameters.


def rtd_classification(features, labels):

    # -------------------------------------- Data preparation ---------------------------------------------------------

    # Rtd features have to be transposed
    features = {k: v.transpose() for k, v in features.items()}

    # Data formatting:
    global_feature_vectors, label_indices = data_preparation_with_global_features(features, labels)

    # Manual class selection (subset of labels)
    selected_commands = ["go", "no", "off", "on", "stop", "yes"]

    # Filtering formatted data according to the previous commands selection
    global_feature_vectors = filter_data(selected_commands, label_indices, global_feature_vectors)

    # Dataframe generation: not used except for visualizing data in a nice way, could be used in order to check the
    # statistical distributions of the features...
    global_df = dataframe_conversion(global_feature_vectors)
    print(global_df)
    print()

    # ----------------------------------- Classification starts here --------------------------------------------------

    X, y = data_and_labels(global_feature_vectors)

    # Splitting the dataset:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.45, random_state=42)

    # Parameter optimization using k-fold cross validation: estimation of the classifier's parameters that maximise
    # specific metrics/scores
    best_parameters = parameters_estimation(X_train, X_test, y_train, y_test)

    # Print the best parameter's configurations that maximise specific metric/scores
    print("List of the parameters used according to a specific scoring parameter:")
    print(json.dumps(best_parameters, indent=4))
    print()

    # Add a specific parameter to the configuration in order to show iterations
    # for i in best_parameters:
    #     best_parameters[i]['verbose'] = 2
    #     # More info on the 'verbose' paramter:
    #     # The verbosity level: if non zero, progress messages are printed. Above 50, the output is sent to stdout.
    #     # The frequency of the messages increases with the verbosity level. If it more than 10, all iterations are
    #     # reported.
    #     # verbose: 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch

    # Selection of a particular configuration: this is done selecting the relative metric
    selected_score = "accuracy"

    # Final classification with optimized parameters:
    print("SVM Classification:")
    print("Parameters set in order to maximise the %s scoring parameter" % selected_score.upper())

    # Instantiate a new classifier object with best parameters for a selected scoring parameter
    clf = SVC(**best_parameters[selected_score])

    # Alternative: creating a new classifier object with manually selected parameters, not optimized
    # clf = SVC(kernel='rbf', decision_function_shape='ovo', verbose=2)

    # Model training:
    print("Model training:")
    with warnings.catch_warnings():
        warnings.filterwarnings(action="ignore", category=UndefinedMetricWarning)
        # .. your divide-by-zero code ..
        clf.fit(X_train, y_train)

    # Model training:
    print("Making predictions:")
    with warnings.catch_warnings():
        warnings.filterwarnings(action="ignore", category=UndefinedMetricWarning)
        # .. your divide-by-zero code ..
        y_predicted = clf.predict(X_test)

        # Performance evaluation
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_predicted))
        print()

        print(classification_report(y_test, y_predicted))
        print()


def mfcc_classification(features, labels):
    # Data preparation
    matrix_feature_vectors = data_preparation_with_local_features(features, labels)

    classes = [1, 2, 3, 4]

    matrix_feature_vectors = filter_data(classes, matrix_feature_vectors)

    matrix_df = dataframe_conversion(matrix_feature_vectors)
    print(matrix_df)
    print()

    X, y = data_and_labels(matrix_feature_vectors)

    # Split the dataset in two equal parts
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=0)

    # -----------------------------------------------------------------

    best_parameters = parameters_estimation(X_train, X_test, y_train, y_test)
    # best_parameters['verbose'] = 2
    print(best_parameters)
    print()

    # -----------------------------------------------------------------

    # Creating a new classifier object
    clf = SVC(**best_parameters)
    # showing iterations
    # clf = SVC(kernel='rbf', decision_function_shape='ovo', verbose=2)

    # Model training
    clf.fit(X_train, y_train)

    # Making predictions
    y_pred = clf.predict(X_test)

    # Performance evaluation
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print()

    print(classification_report(y_test, y_pred))
    print()


# Used with RDT
def data_preparation_with_global_features(features, labels):
    label_indices = {}
    i = 0
    for item in labels:
        if i > 0 and item in label_indices:
            continue
        else:
            i = i + 1
            label_indices[item] = i

    n_of_features = next(iter(features.values())).size
    n_of_samples = len(features)

    # convert to matrix: pre-allocation used
    dataset_matrix = np.zeros(shape=(n_of_samples, n_of_features + 1))
    last_filled = 0
    for key in features:
        command_class = ''.join([i for i in key if not i.isdigit()])
        command_index = label_indices[command_class]
        value = features[key].flatten()
        dataset_matrix[last_filled:last_filled + 1, 0:-1] = value
        dataset_matrix[last_filled:last_filled + 1, -1] = command_index
        last_filled += 1

    return dataset_matrix, label_indices


# Used with MFCC
def data_preparation_with_local_features(features, labels):
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
        dataset_matrix[last_filled:last_filled + value.shape[0], 0:-1] = value
        dataset_matrix[last_filled:last_filled + value.shape[0], -1] = command_index
        # This works only in the case of fixed n_of_segments:
        # t = 0
        # dataset_matrix[value.shape[0]*t:value.shape[0]*(t+1), 0:-1] = value
        # dataset_matrix[value.shape[0]*t:value.shape[0]*(t+1), -1] = command_index
        # t += 1
        last_filled += value.shape[0]

    return dataset_matrix


def filter_data(selected_commands, label_indices, data):
    filtered_data = np.zeros(data.shape[1])
    selected_indices = [label_indices[x] for x in selected_commands]
    for i in range(data.shape[0]):
        if data[i, -1] in selected_indices:
            filtered_data = np.vstack([filtered_data, data[i, :]])

    filtered_data = filtered_data[1:, :]
    return filtered_data


def dataframe_conversion(data):
    # conversion to data frame:
    # add a first row with titles or convert into a Pandas data-frame
    titles = ["feature" + str(i + 1) for i in np.arange(data.shape[1])]
    titles[-1] = "label"
    # print(titles)
    # data_with_titles = np.vstack([titles, data])
    df = pd.DataFrame(data, columns=titles)
    return df


def data_and_labels(data):
    X = data[:, :-1]
    y = data[:, -1]
    return X, y


def parameters_estimation(X_train, X_test, y_train, y_test):
    # Basically, the kernel SVM projects the non-linearly separable data lower dimensions to linearly separable data in
    # higher dimensions in such a way that data points belonging to different classes are allocated to different
    # dimensions.
    #
    # RBF parameters - the effect of the parameters gamma and C of the Radial Basis Function (RBF) kernel SVM:
    # Intuitively, the gamma parameter defines how far the influence of a single training example reaches, with low
    # values meaning ‘far’ and high values meaning ‘close’. The gamma parameters can be seen as the inverse of the
    # radius of influence of samples selected by the model as support vectors.
    # The C parameter trades off correct classification of training examples against maximization of the decision
    # function’s margin. For larger values of C, a smaller margin will be accepted if the decision function is better
    # at classifying all training points correctly. A lower C will encourage a larger margin, therefore a simpler
    # decision function, at the cost of training accuracy. In other words C behaves as a regularization parameter in
    # the SVM.

    # Multi-class classification:
    # SVC and NuSVC implement the “one - versus - one” approach for multi-class classification.
    # In total, n_classes * (n_classes - 1) / 2 classifiers are constructed and each one trains data from two classes.
    # To provide a consistent interface with other classifiers, the decision_function_shape option allows to
    # monotonically transform the results of the “one-versus-one” classifiers to a “one-vs-rest” decision function of
    # shape (n_samples, n_classes).

    # Set the parameters by cross-validation
    tuned_parameters = [
        {'kernel': ['rbf'],
         'gamma': ['scale', 'auto'],  # where 'auto' is the value 1 / n_features
         # by default gamma is set to 'scale', i.e. 1 / (n_features * X.var())
         'C': [1, 10, 100, 1000],  # by default C is set to # 1
         'decision_function_shape': ['ovo', 'ovr']},  # by default this is set to ovr

        {'kernel': ['linear'],
         'C': [1, 10, 100, 1000],
         'decision_function_shape': ['ovo', 'ovr']}
    ]

    scores = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']

    # Creates a dictionary with best parameters for every scoring parameter considered
    best_parameters = {key: None for key in scores}

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(
            SVC(), tuned_parameters, scoring=score, cv=5
        )

        # ignore divide-by-zero warnings, these occur inevitably in the parameter estimation phase and are annoying
        with warnings.catch_warnings():
            warnings.filterwarnings(action="ignore", category=UndefinedMetricWarning)
            # .. your divide-by-zero code ..
            clf.fit(X_train, y_train)

        # print("Best parameters set found on development set:")
        # print()
        # print(clf.best_params_)
        # print()
        # print("Grid scores on development set:")
        # print()
        # means = clf.cv_results_['mean_test_score']
        # stds = clf.cv_results_['std_test_score']
        # for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        #     print("%r: %0.3f (+/-%0.03f) for %r"
        #           % (score, mean, std * 2, params))
        # print()

        # print("Detailed classification report:")
        # print()
        # print("The model is trained on the full development set.")
        # print("The scores are computed on the full evaluation set.")
        # print()

        # ignore divide-by-zero warnings, these occur inevitably in the parameter estimation phase and are annoying
        # with warnings.catch_warnings():
        #     warnings.filterwarnings(action="ignore", category=UndefinedMetricWarning)
        #     # .. your divide-by-zero code ..
        #     y_true, y_pred = y_test, clf.predict(X_test)
        #     print(classification_report(y_true, y_pred))
        # print()

        # print(clf.best_params_)
        best_parameters[score] = clf.best_params_

    return best_parameters


# Not used -------------------------------------------------------------------------------


def plot_stuff(X, y):
    X = X
    Y = y.astype(int)

    # make it binary classification problem
    X = X[np.logical_or(Y == Y[0], Y == Y[1])]
    Y = Y[np.logical_or(Y == Y[0], Y == Y[1])]

    model = SVC(kernel='linear')
    clf = model.fit(X, Y)

    # The equation of the separating plane is given by all x so that np.dot(svc.coef_[0], x) + b = 0.
    # Solve for w3 (z)
    z = lambda x, y: (-clf.intercept_[0] - clf.coef_[0][0] * x - clf.coef_[0][1] * y) / clf.coef_[0][2]

    tmp = np.linspace(-5, 5, 30)
    x, y = np.meshgrid(tmp, tmp)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot3D(X[Y == Y[0], Y[0]], X[Y == Y[0], Y[1]], X[Y == Y[0], Y[2]], 'ob')
    ax.plot3D(X[Y == Y[1], Y[0]], X[Y == Y[1], Y[1]], X[Y == Y[1], Y[2]], 'sr')
    ax.plot_surface(x, y, z(x, y))
    ax.view_init(30, 60)
    plt.show()


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
        train_dataset, test_dataset = old_data_split(features, train_size=0.7)
        fm.create_file(train_name, train_dataset)
        fm.create_file(test_name, test_dataset)
    return fm.read_file(train_file), fm.read_file(test_file)
