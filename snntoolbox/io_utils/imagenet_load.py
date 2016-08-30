"""ImageNet utilities.

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
import json
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


def reorganize_validation(val_path, val_label_path, class_idx_path):
    """Reorganize Validation dataset into folder.

    Parameters
    ----------
    val_path : string
        The path to validation data.
    val_label_path : string
        The label file for validation data.
    class_idx_path : string
        The json file for model index definition.
    """
    if not os.path.isdir(val_path):
        raise ValueError("The Validation Path is not existed!")
    if not os.path.isfile(val_label_path):
        raise ValueError("The validation label file is not existed!")
    if not os.path.isfile(class_idx_path):
        raise ValueError("The class idx file is not existed!")
    class_idx = json.load(open(class_idx_path, "r"))
    label = np.loadtxt(val_label_path, dtype=int)

    for idx in xrange(label.shape[0]):
        img_fn = join(val_path, "ILSVRC2012_val_%08d.JPEG" % (idx+1))

        img_label = label[idx]
        img_label_name = class_idx[str(img_label-1)][0]

        label_path = join(val_path, img_label_name)
        if not os.path.isdir(label_path):
            os.makedirs(label_path)

        shutil.move(img_fn, label_path)
        print ("[MESSAGE] Image %s is moved to %s" % (img_fn, label_path))

    print ("[MESSAGE] The validation data is reorganized.")


def get_imagenet(train_path, test_path, save_path, class_idx_path,
                 filename=None):
    """Load imagenet classification dataset.

    Values are normalized and saved as ``float32`` type. Class vectors are
    converted to binary class matrices. Output can be flattened for use in
    fully-connected networks.

    Parameters
    ----------

    train_path : string
        The path of training data
    test_path : string
        The path of testing data (using validation data)
    save_path : string
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
    if not os.path.isdir(train_path):
        raise ValueError("Training dataset is not found!")
    if not os.path.isdir(test_path):
        raise ValueError("Testing dataset is not found!")
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    if not os.path.isfile(class_idx_path):
        raise ValueError("The class idx file is not existed!")

    class_idx = json.load(open(class_idx_path, "r"))

    classes = []
    for idx in xrange(len(class_idx)):
        classes.append(class_idx[str(idx)][0])

    datagen = ImageDataGenerator()
    train_dataflow = datagen.flow_from_directory(train_path,
                                                 target_size=(224, 224),
                                                 classes=classes,
                                                 batch_size=1000)
    X_train, Y_train = train_dataflow.next()

    X_train[:, 0, :, :] -= 103.939
    X_train[:, 1, :, :] -= 116.779
    X_train[:, 2, :, :] -= 123.68
    X_train = X_train[:, ::-1, :, :]
    # X_train /= 255.

    test_dataflow = datagen.flow_from_directory(test_path,
                                                target_size=(224, 224),
                                                classes=classes,
                                                batch_size=1000)

    X_test, Y_test = test_dataflow.next()

    X_test[:, 0, :, :] -= 103.939
    X_test[:, 1, :, :] -= 116.779
    X_test[:, 2, :, :] -= 123.68
    X_test = X_test[:, ::-1, :, :]
    # X_test /= 255.

    if filename is None:
        filename = ''
    filepath = os.path.join(save_path, filename)
    np.savez_compressed(filepath + 'X_norm', X_train.astype('float32'))
    np.savez_compressed(filepath + 'Y_norm', Y_train.astype('float32'))
    np.savez_compressed(filepath + 'X_test', X_test.astype('float32'))
    np.savez_compressed(filepath + 'Y_test', Y_test.astype('float32'))

if __name__ == '__main__':
    # sample_imagenet("/home/duguyue100/imagenet/ILSVRC2015/Data/CLS-LOC/train/",
    #                 "/home/duguyue100/imagenet_train/", num_samples=50)
    # reorganize_validation(
    #     "/home/duguyue100/data/ILSVRC2012_img_val",
    #     "/home/duguyue100/data/ILSVRC2014_devkit/data/"
    #     "ILSVRC2014_clsloc_validation_ground_truth.txt",
    #     "/home/duguyue100/.keras/models/imagenet_class_index.json")

    get_imagenet('/home/duguyue100/imagenet_train',
                 '/home/duguyue100/data/ILSVRC2012_img_val',
                 '/home/duguyue100/data',
                 '/home/duguyue100/.keras/models/imagenet_class_index.json')
