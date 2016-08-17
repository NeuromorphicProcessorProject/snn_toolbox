# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 12:55:20 2016

@author: rbodo
"""

# For compatibility with python2
from __future__ import print_function, unicode_literals
from __future__ import division, absolute_import
from future import standard_library

import os
import numpy as np
from PIL import Image
from snntoolbox.io_utils.load import to_categorical

standard_library.install_aliases()


def get_facedetection(sourcepath, imagepath, targetpath=None, filename=None):
    """
    Load facedetection dataset.

    Values are normalized and saved as ``float32`` type. Class vectors are
    converted to binary class matrices. Output can be flattened for use in
    fully-connected networks.

    Parameters
    ----------

    sourcepath: string
        Where to find text file containing the filenames and labels of the
        image samples.
    imagepath: string
        Path to image folder.
    targetpath: string, optional
        If a ``path`` is given, the loaded and modified dataset is saved to
        ``path`` directory.
    filename: string, optional
        If a ``path`` is given, the dataset will be written to ``filename``.
        If ``filename`` is not specified, use ``facedetection`` or
        ``facedetection_flat``.

    Returns
    -------

    The dataset as a tuple containing the training and test sample arrays
    (X_train, Y_train, X_test, Y_test).
    With data of the form (channels, num_rows, num_cols), ``X_train`` and
    ``X_test`` have dimension (num_samples, channels, num_rows, num_cols).
    ``Y_train`` and ``Y_test`` have dimension (num_samples, num_classes).

    """

    nb_classes = 2

    # X contains only paths to images. y contains the true labels as integers.
    (X_train, y_train) = load_paths_from_files(sourcepath, imagepath,
                                               'train_36x36.txt')
    (X_test, y_test) = load_paths_from_files(sourcepath, imagepath,
                                             'test_36x36.txt')
    X_train = load_samples(X_train)
    X_test = load_samples(X_test)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    # convert class vectors to binary class matrices
    Y_train = to_categorical(y_train, nb_classes)
    Y_test = to_categorical(y_test, nb_classes)

    if targetpath is not None:
        if filename is None:
            filename = ''
        filepath = os.path.join(targetpath, filename)
        np.savez_compressed(filepath + 'X_norm', X_train)
        np.savez_compressed(filepath + 'X_test', X_test)
        np.savez_compressed(filepath + 'Y_train', Y_train)
        np.savez_compressed(filepath + 'Y_test', Y_test)

    return (X_train, Y_train, X_test, Y_test)


def load_paths_from_files(sourcepath, imagepath, filename):
    filepath = os.path.join(sourcepath, filename)
    assert os.path.isfile(filepath)
    X = []
    Y = []
    for s in np.loadtxt(filepath, dtype=np.str):
        X.append(os.path.join(imagepath, s[0][2:-1]))
        Y.append(int(s[1][2:-1]))
    return np.array(X), np.array(Y)


def load_samples(filepaths, nb_samples=None):
    if nb_samples is None:
        nb_samples = len(filepaths)

    # determine height / width
    img = Image.open(filepaths[0])
    (width, height) = img.size

    chnls = 3 if 'RGB' in img.mode else 1

    # allocate memory
    sample_data = np.zeros((nb_samples, chnls, height, width), dtype="uint8")

    for i in range(nb_samples):
        img = Image.open(filepaths[i])
        if 'RGB' in img.mode:
            r, g, b = img.convert('RGB').split()  # e.g. from RGBA
            sample_data[i, 0, :, :] = np.array(r)
            sample_data[i, 1, :, :] = np.array(g)
            sample_data[i, 2, :, :] = np.array(b)
        else:
            sample_data[i] = np.array(img)

    return sample_data


if __name__ == '__main__':
    sourcepath = '/mnt/2646BAF446BAC3B9/.snntoolbox/datasets/facedetection/' +\
                 'Databases/All_combined/txt'
    imagepath = os.path.abspath(os.path.join(sourcepath, '..', 'images_36x36'))
    targetpath = '/mnt/2646BAF446BAC3B9/.snntoolbox/datasets/facedetection/'
    get_facedetection(sourcepath, imagepath, targetpath)
