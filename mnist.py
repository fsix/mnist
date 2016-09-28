"""FSix MNIST

Usage
    mnist.py train

Options
    -b --bagging    Applies bootstrap aggregation to results of classifications
    -p --perturb    Perturbs the data randomly before training
"""

from docopt import docopt
from mnist import MNIST
from sklearn import metrics
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoding
from utils.bagging import bagging
from utils.perturbations import perturb

import numpy as np


def load_dataset() -> ((np.ndarray, np.ndarray), (np.ndarray, np.ndarray)):
    """Load dataset, taken from CS189 homework template code."""
    mndata = MNIST('./data/')
    X_train, labels_train = map(np.array, mndata.load_training())
    X_test, labels_test = map(np.array, mndata.load_testing())
    X_train = X_train/255.0
    X_test = X_test/255.0
    return (X_train, labels_train), (X_test, labels_test)


def main(arguments: dict) -> None:
    """Runs necessary modules given command-line arguments.

    :param arguments: Command-line arguments
    """
    (X_train, labels_train), (X_test, labels_test) = load_dataset()
    if arguments['--perturb']:
        X_train, labels_train = perturb(X_train, labels_train)
    y_train = OneHotEncoding(labels_train)
    y_test = OneHotEncoding(labels_test)

    classifier = Ridge()
    model = classifier.fit(X_train, y_train)
    pred_labels_train = model.predict(X_train)
    pred_labels_test = model.predict(X_test)

    train_accuracy = metrics.accuracy_score(labels_train, pred_labels_train)
    test_accuracy = metrics.accuracy_score(labels_test, pred_labels_test)
    print('train accuracy:', train_accuracy)
    print('test accuracy:', test_accuracy)

if __name__ == '__main__':
    arguments = docopt(__doc__, version='FSix MNIST 1.0')
    main(arguments)
