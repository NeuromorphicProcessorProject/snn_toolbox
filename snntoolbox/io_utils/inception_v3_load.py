# coding=utf-8

"""ImageNet utilities.

Created on Mon Jun  6 12:55:20 2016

@author: rbodo
"""

# For compatibility with python2
from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import os
from typing import Optional

import numpy as np
from future import standard_library
from keras.preprocessing.image import ImageDataGenerator

standard_library.install_aliases()


def get_imagenet(train_path, test_path, save_path, filename=None):
    """Load imagenet classification dataset.

    Values are normalized and saved as ``float32`` type. Class vectors are
    converted to binary class matrices.

    Three compressed files ``path/filename_x_norm.npz``,
    ``path/filename_x_test.npz``, and ``path/filename_y_test.npz``.
    With data of the form (channels, num_rows, num_cols), ``x_norm`` and
    ``x_test`` have dimension (num_samples, channels, num_rows, num_cols).
    ``y_test`` has dimension (num_samples, num_classes).

    Parameters
    ----------

    train_path : str
        The path of training data
    test_path : str
        The path of testing data (using validation data)
    save_path : str
        If a ``path`` is given, the loaded and modified dataset is saved to
        ``path`` directory.
    filename: Optional[str]
        Basename of file to create. Individual files will be appended
        ``_x_norm``, ``_x_test``, etc.
    """

    if not os.path.isdir(train_path):
        raise ValueError("Training dataset not found!")
    if not os.path.isdir(test_path):
        raise ValueError("Testing dataset not found!")
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    target_size = (299, 299)
    num_norm_samples = 10
    num_test_samples = 5000

    datagen = ImageDataGenerator()
    train_dataflow = datagen.flow_from_directory(train_path,
                                                 target_size=target_size,
                                                 batch_size=num_norm_samples)
    x_train, y_train = train_dataflow.next()

    x_train /= 255.
    x_train -= 0.5
    x_train *= 2.

    test_dataflow = datagen.flow_from_directory(test_path,
                                                target_size=target_size,
                                                batch_size=num_test_samples)
    x_test, y_test = test_dataflow.next()

    x_test /= 255.
    x_test -= 0.5
    x_test *= 2.

    if filename is None:
        filename = ''
    filepath = os.path.join(save_path, filename)
    np.savez_compressed(filepath + 'x_norm', x_train.astype('float32'))
    np.savez_compressed(filepath + 'x_test', x_test.astype('float32'))
    np.savez_compressed(filepath + 'y_test', y_test.astype('float32'))

if __name__ == '__main__':
    trainpath = '/home/rbodo/.snntoolbox/Datasets/imagenet/training'
    testpath = '/home/rbodo/.snntoolbox/Datasets/imagenet/validation'
    savepath = '/home/rbodo/.snntoolbox/Datasets/imagenet/inception'
    get_imagenet(trainpath, testpath, savepath)
