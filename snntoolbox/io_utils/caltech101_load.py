# -*- coding: utf-8 -*-
"""
Helper functions to load, process, augment and save Caltech101 dataset.

Created on Mon Jun  6 12:55:20 2016

@author: rbodo
"""

# For compatibility with python2
from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import os

import numpy as np
from future import standard_library
from keras.preprocessing.image import ImageDataGenerator

standard_library.install_aliases()


def get_caltech101(path, filename=None):
    """Get caltech101 classification dataset.

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
        Basename of file to create. Individual files will be appended
        ``_x_norm``, ``_x_test``, etc.

    Returns
    -------

    Three compressed files ``path/filename_x_norm.npz``,
    ``path/filename_x_test.npz``, and ``path/filename_y_test.npz``.
    With data of the form (channels, num_rows, num_cols), ``x_norm`` and
    ``x_test`` have dimension (num_samples, channels, num_rows, num_cols).
    ``y_test`` has dimension (num_samples, num_classes).
    """

    num_samples = 9144
    target_size = (180, 240)

    datagen = ImageDataGenerator(rescale=1./255,
                                 featurewise_center=True,
                                 featurewise_std_normalization=True)

    # Compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied)
    x = ImageDataGenerator(rescale=1./255).flow_from_directory(
        path, target_size, batch_size=num_samples).next()[0]
    datagen.fit(x)

    dataflow = datagen.flow_from_directory(
        path, target_size, batch_size=num_samples)

    x_test, y_test = dataflow.next()

    if filename is None:
        filename = ''
    filepath = os.path.join(path, filename)
    np.savez_compressed(filepath + 'x_norm', x_test[::100].astype('float32'))
    np.savez_compressed(filepath + 'x_test', x_test.astype('float32'))
    np.savez_compressed(filepath + 'y_test', y_test)

if __name__ == '__main__':
    get_caltech101('/home/rbodo/.snntoolbox/Datasets/caltech101/original/')
