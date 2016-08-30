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
from os.path import join
import shutil
import random
import numpy as np

from keras.preprocessing.image import ImageDataGenerator

standard_library.install_aliases()


def sample_imagenet(origin_path, target_path, num_samples=50):
    """Randomly sample images from ImageNet Training Dataset.

    This function ramdomly samples images from each class of ImageNet Training
    Dataset for normalizing pretained ImageNet model.

    In case you don't have to permission to the data folder, make sure
    your are running this function with super user.

    Parameters
    ----------
    origin_path : string
        The path to training dataset
    target_path : string
        The path to target destination
    """
    if not os.path.isdir(origin_path):
        raise ValueError("The source folder is not existed!")
    if not os.path.isdir(target_path):
        os.makedirs(target_path)
        print ("[MESSAGE] WARNING! The target path is not existed, "
               "The path is created automatically.")

    print ("[MESSAGE] Start Copying.")
    folder_list = [f for f in os.listdir(origin_path)
                   if os.path.isdir(join(origin_path, f))]

    for folder_name in folder_list:
        if not os.path.isdir(join(target_path, folder_name)):
            os.makedirs(join(target_path, folder_name))
        folder_path = join(origin_path, folder_name)
        file_list = [f for f in os.listdir(folder_path)
                     if os.path.isfile(join(folder_path, f)) and ".JPEG" in f]

        if num_samples > len(file_list):
            file_idx = range(len(file_list))
        else:
            file_idx = random.sample(range(len(file_list)), num_samples)

        for idx in file_idx:
            shutil.copy(join(folder_path, file_list[idx]),
                        join(target_path, folder_name))
            print ("[MESSAGE] Image %s is copied to %s" %
                   (file_list[idx], join(target_path, folder_name)))

    print ("[MESSAGE] Images are sampled! Stored at %s" % (target_path))


def get_imagenet(path, filename=None):
    """Load imagenet classification dataset.

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
    datagen = ImageDataGenerator()
    dataflow = datagen.flow_from_directory(path, target_size=(224, 224),
                                           batch_size=1000)
    X_test, Y_test = dataflow.next()

#    X_test[:, 0, :, :] -= 103.939
#    X_test[:, 1, :, :] -= 116.779
#    X_test[:, 2, :, :] -= 123.68
#    # 'RGB'->'BGR'
#    X_test = X_test[:, ::-1, :, :]
#    X_test /= 255.

    y_shape = list(Y_test.shape)
    y_shape[-1] = 1000
    print(y_shape)
    y_test = np.zeros(y_shape)
    y_test[:, :102] = Y_test

    if filename is None:
        filename = ''
    filepath = os.path.join(path, filename)
#    np.savez_compressed(filepath + 'X_norm', X_test[::100].astype('float32'))
#    np.savez_compressed(filepath + 'X_test', X_test.astype('float32'))
    np.savez_compressed(filepath + 'Y_test', y_test)

if __name__ == '__main__':
    # get_imagenet('/home/rbodo/.snntoolbox/datasets/caltech101/original/')
    sample_imagenet("/home/duguyue100/imagenet/ILSVRC2015/Data/CLS-LOC/train/",
                    "/home/duguyue100/imagenet_train/", num_samples=5)
