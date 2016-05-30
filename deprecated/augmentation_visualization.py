# -*- coding: utf-8 -*-
"""
Show effect of transforming input images (rotation, noise, centered mean, ...)

Created on Fri Dec 18 10:04:22 2015

@author: rbodo
"""

# For compatibility with python2
from __future__ import print_function, unicode_literals
from __future__ import division, absolute_import
from future import standard_library
standard_library.install_aliases()


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    from random import randint
    from keras.preprocessing.image import ImageDataGenerator
    from snntoolbox.io.load import get_cifar10

    (X_train, Y_train, X_test, Y_test) = get_cifar10()

    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=20,  # randomly rotate images in range (0 to 180 deg)
        width_shift_range=0.2,  # randomly shift images horizontally
                                # (fraction of total width)
        height_shift_range=0.2,  # randomly shift images vertically
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    ind = randint(0, X_test.shape[0])
    x = X_test[ind:ind+1, :, :, :]

    # compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied)
    datagen.fit(X_test)

    x_aug = [X_aug for X_aug, Y_aug in datagen.flow(x, Y_test[ind, :])][0]

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(np.transpose(x, axes=(0, 2, 3, 1))[0, :, :, :])
    plt.title('Original Image')
    plt.subplot(1, 2, 2)
    plt.imshow(np.transpose(x_aug, axes=(0, 2, 3, 1))[0, :, :, :])
    plt.title('Augmented Image')
    plt.tight_layout()
