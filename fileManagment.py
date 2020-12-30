# Simple functions to create and to load files


def readFile(file_name):
    try:
        import cPickle as pickle
    except ImportError:  # Python 3.x
        import pickle

    with open(file_name, 'rb') as fp:
        return pickle.load(fp)


def createFile(file_name, content):
    try:
        import cPickle as pickle
    except ImportError:  # Python 3.x
        import pickle
    with open(file_name, 'wb') as fp:
        pickle.dump(content, fp, protocol=pickle.HIGHEST_PROTOCOL)