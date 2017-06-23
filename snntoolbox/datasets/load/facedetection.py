# -*- coding: utf-8 -*-
"""
Helper functions to load, process, augment and save facedetection dataset.

Created on Mon Jun  6 12:55:20 2016

@author: rbodo
"""

# For compatibility with python2
from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import os

import numpy as np
from PIL import Image
from future import standard_library
from snntoolbox.datasets.utils import to_categorical

standard_library.install_aliases()


def get_facedetection(sourcepath, imagepath, targetpath=None, filename=None):
    """Get facedetection dataset.

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
    (x_train, y_train, x_test, y_test).
    With data of the form (channels, num_rows, num_cols), ``x_train`` and
    ``x_test`` have dimension (num_samples, channels, num_rows, num_cols).
    ``y_train`` and ``y_test`` have dimension (num_samples, num_classes).
    """

    nb_classes = 2

    # X contains only paths to images. y contains the true labels as integers.
    (x_train, y_train) = load_paths_from_files(sourcepath, imagepath,
                                               'train_36x36.txt')
    (x_test, y_test) = load_paths_from_files(sourcepath, imagepath,
                                             'test_36x36.txt')
    x_train = load_samples(x_train)
    x_test = load_samples(x_test)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # convert class vectors to binary class matrices
    y_train = to_categorical(y_train, nb_classes)
    y_test = to_categorical(y_test, nb_classes)

    if targetpath is not None:
        if filename is None:
            filename = ''
        filepath = os.path.join(targetpath, filename)
        np.savez_compressed(filepath + 'x_norm', x_train)
        np.savez_compressed(filepath + 'x_test', x_test)
        np.savez_compressed(filepath + 'y_train', y_train)
        np.savez_compressed(filepath + 'y_test', y_test)

    return x_train, y_train, x_test, y_test


def load_paths_from_files(sourcepath, imagepath, filename):
    """Load paths to data samples from a file.

    Parameters
    ----------

    sourcepath: str
        Path to source file
    imagepath: str
        Path to samples
    filename: str
        Name of source file

    Returns
    -------

    : np.array, np.array
        Samples and targets.
    """

    filepath = os.path.join(sourcepath, filename)
    assert os.path.isfile(filepath)
    x = []
    y = []
    for s in np.loadtxt(filepath, dtype=np.str):
        x.append(os.path.join(imagepath, s[0][2:-1]))
        y.append(int(s[1][2:-1]))
    return np.array(x), np.array(y)


def load_samples(filepaths, nb_samples=None):
    """Load samples from file containing the paths to individual samples.

    Parameters
    ----------

    filepaths: np.array
    nb_samples: Optional[int]

    Returns
    -------

    sample_data: np.array
    """

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
    source_path = '/mnt/2646BAF446BAC3B9/.snntoolbox/datasets/facedetection/' +\
                 'Databases/All_combined/txt'
    image_path = os.path.abspath(os.path.join(source_path, '..',
                                              'images_36x36'))
    target_path = '/mnt/2646BAF446BAC3B9/.snntoolbox/datasets/facedetection/'
    get_facedetection(source_path, image_path, target_path)
