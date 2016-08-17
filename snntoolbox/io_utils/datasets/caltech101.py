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

from keras.preprocessing.image import ImageDataGenerator

standard_library.install_aliases()


def get_caltech101(path, filename=None):
    """
    Load caltech101 classification dataset.

    Values are normalized and saved as ``float32`` type. Class vectors are
    converted to binary class matrices. Output can be flattened for use in
    fully-connected networks.

    Parameters
    ----------

    path: string, optional
        If a ``path`` is given, the loaded and modified dataset is saved to
        ``path`` directory.
    filename: string, optional
        Basename of file to create. Individual files will be appended
        ``_X_norm``, ``_X_test``, etc.

    Returns
    -------

    Three compressed files ``path/filename_X_norm.npz``,
    ``path/filename_X_test.npz``, and ``path/filename_Y_test.npz``.
    With data of the form (channels, num_rows, num_cols), ``X_norm`` and
    ``X_test`` have dimension (num_samples, channels, num_rows, num_cols).
    ``Y_test`` has dimension (num_samples, num_classes).

    """

    datagen = ImageDataGenerator(rescale=1./255)
    dataflow = datagen.flow_from_directory(path, target_size=(180, 240),
                                           batch_size=9144)
    X_test, Y_test = dataflow.next()

    if filename is None:
        filename = ''
    filepath = os.path.join(path, filename)
    np.savez_compressed(filepath + 'X_norm', X_test[::100].astype('float32'))
    np.savez_compressed(filepath + 'X_test', X_test.astype('float32'))
    np.savez_compressed(filepath + 'Y_test', Y_test)

if __name__ == '__main__':
    get_caltech101('/home/rbodo/.snntoolbox/datasets/caltech101/original/')
