# coding=utf-8

"""ImageNet utilities.

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


def get_imagenet(train_path, test_path, save_path, filename=None,
                 class_idx_path=None):
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

    class_idx_path :
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
    # train_dataflow = datagen.flow_from_directory(train_path,
    #                                              target_size=target_size,
    #                                              batch_size=num_norm_samples)
    # x_train, y_train = train_dataflow.next()
    #
    # x_train /= 255.
    # x_train -= 0.5
    # x_train *= 2.

    if class_idx_path:
        import json
        class_idx = json.load(open(class_idx_path, "r"))
        classes = [class_idx[str(idx)][0] for idx in range(len(class_idx))]
    else:
        classes = None

    test_dataflow = datagen.flow_from_directory(test_path,
                                                target_size=target_size,
                                                classes=classes,
                                                batch_size=num_test_samples)
    for i, x_test, y_test in enumerate(test_dataflow):

        x_test = np.add(np.multiply(x_test, 2. / 255.), - 1.)

        if filename is None:
            filename = ''
        filepath = os.path.join(save_path, filename)
        step = int(len(x_test) / num_norm_samples)
        np.savez_compressed(filepath + 'x_test' + str(i),
                            x_test.astype('float32'))
        np.savez_compressed(filepath + 'y_test' + str(i),
                            y_test.astype('float32'))
        if i == 0:
            np.savez_compressed(filepath + 'x_norm',
                                x_test[::step].astype('float32'))

if __name__ == '__main__':
    path = '/home/rbodo/.snntoolbox/Datasets/imagenet'
    trainpath = path + '/training'
    testpath = path + '/validation'
    savepath = path + '/GoogLeNet'
    classidxpath = savepath + '/imagenet_class_index.json'
    get_imagenet(trainpath, testpath, savepath, class_idx_path=classidxpath)
