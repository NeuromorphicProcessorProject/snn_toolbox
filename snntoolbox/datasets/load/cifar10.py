# -*- coding: utf-8 -*-
"""
Helper functions to load, process, augment and save Cifar10 dataset.

Created on Mon Jun  6 12:55:10 2016

@author: rbodo
"""

# For compatibility with python2
from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import os

import numpy as np
from future import standard_library
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from snntoolbox.datasets.utils import to_categorical

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

    Three compressed files ``path/filename_x_norm.npz``,
    ``path/filename_x_test.npz``, and ``path/filename_y_test.npz``.
    With data of the form (channels, num_rows, num_cols), ``x_norm`` and
    ``x_test`` have dimension (num_samples, channels*num_rows*num_cols)
    in case ``flat==True``, and (num_samples, channels, num_rows, num_cols)
    otherwise. ``y_test`` has dimension (num_samples, num_classes).

    """

    # Whether to apply global contrast normalization and ZCA whitening
    gcn = False
    zca = False
    nb_classes = 10

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Convert class vectors to binary class matrices
    y_train = to_categorical(y_train, nb_classes)
    y_test = to_categorical(y_test, nb_classes)

    datagen = ImageDataGenerator(rescale=1./255, featurewise_center=gcn,
                                 featurewise_std_normalization=gcn,
                                 zca_whitening=zca)
    datagen.fit(x_test/255.)

    testflow = datagen.flow(x_test, y_test, batch_size=len(x_test))
    x_test, y_test = testflow.next()

    normflow = datagen.flow(x_train, y_train, batch_size=int(len(x_train)/3))
    x_norm, y_norm = normflow.next()

    if flat:
        x_norm = x_norm.reshape(x_norm.shape[0], np.prod(x_norm.shape[1:]))
        x_test = x_test.reshape(x_test.shape[0], np.prod(x_test.shape[1:]))

    if path is not None:
        if not os.path.exists(path):
            os.makedirs(path)
        if filename is None:
            filename = ''
        filepath = os.path.join(path, filename)
        np.savez_compressed(filepath+'x_norm', x_norm.astype('float32'))
        np.savez_compressed(filepath+'x_test', x_test.astype('float32'))
        # np.savez_compressed(filepath+'y_train', y_train.astype('float32'))
        np.savez_compressed(filepath+'y_test', y_test.astype('float32'))

#    return (x_train, y_train, x_test, y_test)

if __name__ == '__main__':
    get_cifar10('/home/rbodo/.snntoolbox/Datasets/cifar10/original/')
