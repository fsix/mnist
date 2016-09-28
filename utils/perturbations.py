"""
Perturbations
Add random perturbations to the dataset to make the classifier more robust.

- idea taken from Google's "Explaining and Harnessing Adversarial Examples"
https://arxiv.org/pdf/1412.6572v3.pdf
"""

import numpy as np
from scipy.ndimage import interpolation


SKEW_MEAN = 0.26
SKEW_SIGMA = 0.01


def perturb(X, y, labels, n=60000, dim=28):
    """Duplicate training data, by perturbing each image several ways.

    Each image will see the following perturbations:
    1. rotation clockwise by 3-7 degrees
    2. rotation counterclockwise by 3-7 degrees
    3. skew by (N(3, 1), N(1, 0.5))
    """
    num_transformations = 1

    print('[Perturb] Preprocessing images...')
    X_new = X.copy()
    X_new = np.concatenate([X_new, _perturb_skew(X, n)])

    print('[Perturb] All samples generated. Shuffling...')

    y_new = np.concatenate([y]*(1+num_transformations))
    labels_new = np.concatenate([labels]*(1+num_transformations))

    print('[Perturb] Preprocessing complete. ({num}x samples)'.format(
        num=num_transformations + 1
    ))
    return X_new.reshape(X_new.shape[0], dim*dim), y_new, labels_new


def _pick(X, n=None):
    """Randomly pick and return n row vectors from X."""
    random = np.random.random_integers(low=0, high=n or X.shape[0], size=n)
    return X[random]


def _perturb_cw(X, n=None):
    """Rotate the images by some randomly-generated number of degrees."""
    print('[Perturb] Rotating images clockwise...')
    return interpolation.rotate(
        _pick(X, n),
        np.random.random()*4+3)


def _perturb_ccw(X, n=None):
    """Rotate the images by some randomly-generated number of degrees."""
    print('[Perturb] Rotating images counter-clockwise...')
    return interpolation.rotate(
        _pick(X, n),
        -(np.random.random()*4+3))


def _perturb_skew(X, n=None):
    """Skew the images."""
    print('[Perturb] Skewing images...')
    X_new = _pick(X, n) if n < X.shape[0] else X.copy()
    for i, image in enumerate(X_new):
        X_new[i] = _skew(image)
    return X_new


def _skew(image):
    """Skew the image provided.

    Taken from StackOverflow:
    http://stackoverflow.com/a/33088550/4855984
    """
    image = image.reshape(28, 28)
    h, l = image.shape
    dl = np.random.normal(loc=SKEW_MEAN, scale=SKEW_SIGMA)

    def mapping(lc):
        l, c = lc
        dec = (dl*(l-h))/h
        return l, c+dec
    return interpolation.geometric_transform(
        image, mapping, (h, l), order=5, mode='nearest').reshape(784)