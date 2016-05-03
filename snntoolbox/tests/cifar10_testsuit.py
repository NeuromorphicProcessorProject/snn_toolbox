# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 14:50:36 2015

@author: rbodo
"""

# For compatibility with python2
from __future__ import print_function, unicode_literals
from __future__ import division, absolute_import
from future import standard_library
standard_library.install_aliases()


if __name__ == '__main__':

    from snntoolbox.config import update_setup
    from snntoolbox.io.load import load_model
    from keras.datasets import cifar10
    import matplotlib.pyplot as plt
    from random import randint
    import numpy as np
    # For compatibility with python2
    from builtins import range

    globalparams = {'dataset': 'cifar10',
                    'architecture': 'cnn',
                    'model_lib': 'lasagne',
                    'filename': 'lasagne'}
    update_setup(globalparams)
    ann = load_model('ann_'+globalparams['filename'])['model']
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    plt.figure(figsize=(17, 9))
    for i in range(10):
        ind = randint(0, X_test.shape[0])
        x = X_test[ind:ind+1, :, :, :]
        plt.subplot(2, 5, i+1)
        plt.imshow(np.transpose(x, axes=(0, 2, 3, 1))[0])
        plt.title('Input Image')
        plt.xlabel('Predicted class: ' +
                   np.str(ann.predict_classes(x, verbose=0)[0]) + '\n' +
                   'ground truth: ' + np.str(y_test[ind, 0]))
