# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 12:55:10 2016

@author: rbodo
"""

# For compatibility with python2
from __future__ import print_function, unicode_literals
from __future__ import division, absolute_import
from future import standard_library

import os
import numpy as np
from keras.datasets import cifar10
from snntoolbox.io_utils.load import to_categorical

standard_library.install_aliases()


def get_cifar10(path=None, filename=None, flat=False):
    """
    Load cifar10 classification dataset.

    Values are normalized and saved as ``float32`` type. Class vectors are
    converted to binary class matrices. Output can be flattened for use in
    fully-connected networks.

    Parameters
    ----------

    path: string, optional
        If a ``path`` is given, the loaded and modified dataset is saved to
        ``path`` directory.
    filename: string, optional
        If a ``path`` is given, the dataset will be written to ``filename``.
        If ``filename`` is not specified, use ``cifar10`` or ``cifar10_flat``.
    flat: Boolean, optional
        If ``True``, the output is flattened. Defaults to ``False``.

    Returns
    -------

    Three compressed files ``path/filename_X_norm.npz``,
    ``path/filename_X_test.npz``, and ``path/filename_Y_test.npz``.
    With data of the form (channels, num_rows, num_cols), ``X_norm`` and
    ``X_test`` have dimension (num_samples, channels*num_rows*num_cols)
    in case ``flat==True``, and (num_samples, channels, num_rows, num_cols)
    otherwise. ``Y_test`` has dimension (num_samples, num_classes).

    """

    nb_classes = 10

    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    # convert class vectors to binary class matrices
    Y_train = to_categorical(y_train, nb_classes)
    Y_test = to_categorical(y_test, nb_classes)

    if flat:
        X_train = X_train.reshape(X_train.shape[0], np.prod(X_train.shape[1:]))
        X_test = X_test.reshape(X_test.shape[0], np.prod(X_test.shape[1:]))

    if path is not None:
        if filename is None:
            filename = 'cifar10_flat_' if flat else ''
        filepath = os.path.join(path, filename)
        np.savez_compressed(filepath+'X_norm', X_train)
        np.savez_compressed(filepath+'X_test', X_test)
#       np.savez_compressed(filepath+'Y_train', Y_train)
        np.savez_compressed(filepath+'Y_test', Y_test)

    return (X_train, Y_train, X_test, Y_test)
