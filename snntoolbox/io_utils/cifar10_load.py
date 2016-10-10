# -*- coding: utf-8 -*-
"""
Helper functions to load, process, augment and save Cifar10 dataset.

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
from snntoolbox.io_utils.common import to_categorical
from keras.preprocessing.image import ImageDataGenerator

standard_library.install_aliases()


def get_cifar10(path=None, filename=None, flat=False):
    """Get cifar10 classification dataset.

    Values are normalized and saved as ``float32`` type. Class vectors are
    converted to binary class matrices. Output can be flattened for use in
    fully-connected networks. Can perform preprocessing using a Keras
    ImageDataGenerator.

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

    # Whether to apply global contrast normalization and ZCA whitening
    gcn = False
    zca = False
    nb_classes = 10

    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    # Convert class vectors to binary class matrices
    Y_train = to_categorical(y_train, nb_classes)
    Y_test = to_categorical(y_test, nb_classes)

    datagen = ImageDataGenerator(rescale=1./255, featurewise_center=gcn,
                                 featurewise_std_normalization=gcn,
                                 zca_whitening=zca)
    datagen.fit(X_test/255.)

    testflow = datagen.flow(X_test, Y_test, batch_size=len(X_test))
    X_test, Y_test = testflow.next()

    normflow = datagen.flow(X_train, Y_train, batch_size=int(len(X_train)/3))
    X_norm, Y_norm = normflow.next()

    if flat:
        X_norm = X_norm.reshape(X_norm.shape[0], np.prod(X_norm.shape[1:]))
        X_test = X_test.reshape(X_test.shape[0], np.prod(X_test.shape[1:]))

    if path is not None:
        if filename is None:
            filename = ''
        filepath = os.path.join(path, filename)
        np.savez_compressed(filepath+'X_norm', X_norm.astype('float32'))
        np.savez_compressed(filepath+'X_test', X_test.astype('float32'))
#       np.savez_compressed(filepath+'Y_train', Y_train.astype('float32'))
        np.savez_compressed(filepath+'Y_test', Y_test.astype('float32'))

#    return (X_train, Y_train, X_test, Y_test)

if __name__ == '__main__':
    get_cifar10('/home/rbodo/.snntoolbox/Datasets/cifar10/original/')
