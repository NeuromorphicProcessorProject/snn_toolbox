# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 12:54:49 2016

@author: rbodo
"""

# For compatibility with python2
from __future__ import print_function, unicode_literals
from __future__ import division, absolute_import
from future import standard_library
from builtins import open

import sys
import os
import gzip
import numpy as np
from six.moves import cPickle
from snntoolbox.io_utils.common import download_dataset, to_categorical

standard_library.install_aliases()


def get_mnist(path=None, filename=None, flat=False):
    """
    Load mnist classification dataset.

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
        If ``filename`` is not specified, use ``mnist`` or ``mnist_flat``.
    flat: Boolean, optional
        If ``True``, the output is flattened. Defaults to ``False``.

    Returns
    -------

    The dataset as a tuple containing the training and test sample arrays
    (X_train, Y_train, X_test, Y_test).
    With data of the form (channels, num_rows, num_cols), ``X_train`` and
    ``X_test`` have dimension (num_samples, channels*num_rows*num_cols)
    in case ``flat==True``, and
    (num_samples, channels, num_rows, num_cols) otherwise.
    ``Y_train`` and ``Y_test`` have dimension (num_samples, num_classes).

    """

    nb_classes = 10

    d = download_dataset(
        'mnist.pkl.gz',
        origin='https://s3.amazonaws.com/img-datasets/mnist.pkl.gz')

    if d.endswith('.gz'):
        f = gzip.open(d, 'rb')
    else:
        f = open(d, 'rb')

    if sys.version_info < (3,):
        (X_train, y_train), (X_test, y_test) = cPickle.load(f)
    else:
        (X_train, y_train), (X_test, y_test) = cPickle.load(f,
                                                            encoding='bytes')
    f.close()

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    # Convert class vectors to binary class matrices
    Y_train = to_categorical(y_train, nb_classes)
    Y_test = to_categorical(y_test, nb_classes)

    # Data container has no channel dimension, but we need 4D input for CNN:
    if X_train.ndim < 4 and not flat:
        X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1],
                                  X_train.shape[2])
        X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1],
                                X_test.shape[2])

    if flat:
        X_train = X_train.reshape(X_train.shape[0], np.prod(X_train.shape[1:]))
        X_test = X_test.reshape(X_test.shape[0], np.prod(X_test.shape[1:]))

    if path is not None:
        if filename is None:
            filename = ''
        filepath = os.path.join(path, filename)
        np.savez_compressed(filepath+'X_norm', X_train)
        np.savez_compressed(filepath+'X_test', X_test)
#       np.savez_compressed(filepath+'Y_train', Y_train)
        np.savez_compressed(filepath+'Y_test', Y_test)

    return (X_train, Y_train, X_test, Y_test)
