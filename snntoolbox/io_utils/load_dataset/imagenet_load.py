# coding=utf-8

"""ImageNet utilities.

Created on Mon Jun  6 12:55:20 2016

@author: rbodo
"""

# For compatibility with python2
from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import json
import os
import random
import shutil
from os.path import join
from collections import OrderedDict

import numpy as np
from future import standard_library
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

    origin_path: str
        The path to training dataset
    target_path: str
        The path to target destination
    num_samples: Optional[int]
        Number of samples to get from ImageNet
    """

    if not os.path.isdir(origin_path):
        raise ValueError("The source folder is not existed!")
    if not os.path.isdir(target_path):
        os.makedirs(target_path)
        print("[MESSAGE] WARNING! The target path is not existed, "
              "The path is created automatically.")

    print("[MESSAGE] Start Copying.")
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
            print("[MESSAGE] Image %s is copied to %s" %
                  (file_list[idx], join(target_path, folder_name)))

    print("[MESSAGE] Images are sampled! Stored at %s" % target_path)


def generate_class_idx(class_map_path, save_path, filename=None):
    """Generate class index mapping.

    Parameters
    ----------
    class_map_path : string
        class mapping file
    save_path : string
        the destination of the mapping file in json.
    filename : string
        if it's None, then the filename is assigned as imagenet_class_map.json
    """
    if not os.path.isfile(class_map_path):
        raise ValueError("The class mapping file is not available!")
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
        print("[MESSAGE] WARNING! The target path is not existed, "
              "The path is created automatically.")

    if filename is None:
        filename = "imagenet_class_map.json"

    file_path = join(save_path, filename)

    with open(class_map_path) as f:
        class_map_list = [line.split() for line in f]

    cls_dict = OrderedDict()
    for cls_item in class_map_list:
        cls_dict[str(int(cls_item[1])-1)] = [cls_item[0], cls_item[2]]

    with open(file_path, mode="w") as f_out:
        json.dump(cls_dict, f_out)
    print("[MESSAGE] The class mapping file is dumped at %s." % file_path)


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

    for idx in range(label.shape[0]):
        img_fn = join(val_path, "ILSVRC2012_val_%08d.JPEG" % (idx+1))

        img_label = label[idx]
        img_label_name = class_idx[str(img_label-1)][0]

        label_path = join(val_path, img_label_name)
        if not os.path.isdir(label_path):
            os.makedirs(label_path)

        shutil.move(img_fn, label_path)
        print("[MESSAGE] Image %s is moved to %s" % (img_fn, label_path))

    print("[MESSAGE] The validation data is reorganized.")


def get_imagenet(train_path, test_path, save_path, class_idx_path,
                 filename=None):
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
    class_idx_path: str
        Path to class indexes
    filename: Optional[str]
        Basename of file to create. Individual files will be appended
        ``_x_norm``, ``_x_test``, etc.
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
    for idx in range(len(class_idx)):
        classes.append(class_idx[str(idx)][0])

    datagen = ImageDataGenerator()
    train_dataflow = datagen.flow_from_directory(train_path,
                                                 target_size=(224, 224),
                                                 classes=classes,
                                                 batch_size=10)
    x_train, y_train = train_dataflow.next()

    x_train[:, 0, :, :] -= 103.939
    x_train[:, 1, :, :] -= 116.779
    x_train[:, 2, :, :] -= 123.68
    x_train = x_train[:, ::-1, :, :]
    # x_train /= 255.

    test_dataflow = datagen.flow_from_directory(test_path,
                                                target_size=(224, 224),
                                                classes=classes,
                                                batch_size=10000)

    x_test, y_test = test_dataflow.next()

    x_test[:, 0, :, :] -= 103.939
    x_test[:, 1, :, :] -= 116.779
    x_test[:, 2, :, :] -= 123.68
    x_test = x_test[:, ::-1, :, :]
    # x_test /= 255.

    if filename is None:
        filename = ''
    filepath = os.path.join(save_path, filename)
    np.savez_compressed(filepath + 'x_norm', x_train.astype('float32'))
    np.savez_compressed(filepath + 'y_norm', y_train.astype('float32'))
    np.savez_compressed(filepath + 'x_test', x_test.astype('float32'))
    np.savez_compressed(filepath + 'y_test', y_test.astype('float32'))

if __name__ == '__main__':
    # sample_imagenet("/home/duguyue100/imagenet/ILSVRC2015/Data/CLS-LOC/train/",
    #                 "/home/duguyue100/imagenet_train/", num_samples=50)
    # reorganize_validation(
    #     "/home/duguyue100/data/ILSVRC2012_img_val",
    #     "/home/duguyue100/data/ILSVRC2014_devkit/data/"
    #     "ILSVRC2014_clsloc_validation_ground_truth.txt",
    #     "/home/duguyue100/.keras/models/imagenet_class_map.json")

    get_imagenet('/home/duguyue100/imagenet_train',
                 '/home/duguyue100/data/ILSVRC2012_img_val',
                 '/home/duguyue100/data',
                 '/home/duguyue100/.keras/models/imagenet_class_index.json')

    # generate_class_idx("/home/duguyue100/.keras/models/map_clsloc.txt",
    #                    "/home/duguyue100/.keras/models/", filename=None)
