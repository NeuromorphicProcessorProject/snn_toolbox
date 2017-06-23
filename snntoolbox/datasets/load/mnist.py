# -*- coding: utf-8 -*-
"""
Helper functions to load, process, augment and save MNIST dataset.

Created on Mon Jun  6 12:54:49 2016

@author: rbodo
"""

# For compatibility with python2
from __future__ import print_function, unicode_literals
from __future__ import division, absolute_import
from future import standard_library

import sys
import os
import gzip
import numpy as np
# noinspection PyUnresolvedReferences
from six.moves import cPickle
from keras.datasets.mnist import load_data
from snntoolbox.datasets.utils import to_categorical

standard_library.install_aliases()


def get_mnist(path=None, filename=None, flat=False):
    """Get mnist classification dataset.

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
    (x_train, y_train, x_test, y_test).
    With data of the form (channels, num_rows, num_cols), ``x_train`` and
    ``x_test`` have dimension (num_samples, channels*num_rows*num_cols)
    in case ``flat==True``, and
    (num_samples, channels, num_rows, num_cols) otherwise.
    ``y_train`` and ``y_test`` have dimension (num_samples, num_classes).

    """

    nb_classes = 10

    d = load_data()

    if d.endswith('.gz'):
        f = gzip.open(d, 'rb')
    else:
        f = open(d, 'rb')

    if sys.version_info < (3,):
        (x_train, y_train), (x_test, y_test) = cPickle.load(f)
    else:
        (x_train, y_train), (x_test, y_test) = cPickle.load(f,
                                                            encoding='bytes')
    f.close()

    x_train /= 255
    x_test /= 255

    # Convert class vectors to binary class matrices
    y_train = to_categorical(y_train, nb_classes)
    y_test = to_categorical(y_test, nb_classes)

    # Data container has no channel dimension, but we need 4D input for CNN:
    if x_train.ndim < 4 and not flat:
        x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1],
                                  x_train.shape[2])
        x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1],
                                x_test.shape[2])

    if flat:
        x_train = x_train.reshape(x_train.shape[0], np.prod(x_train.shape[1:]))
        x_test = x_test.reshape(x_test.shape[0], np.prod(x_test.shape[1:]))

    if path is not None:
        if filename is None:
            filename = ''
        filepath = os.path.join(path, filename)
        np.savez_compressed(filepath+'x_norm', x_train.astype('float32'))
        np.savez_compressed(filepath+'x_test', x_test.astype('float32'))
#       np.savez_compressed(filepath+'y_train', y_train.astype('float32'))
        np.savez_compressed(filepath+'y_test', y_test.astype('float32'))

    return x_train, y_train, x_test, y_test
