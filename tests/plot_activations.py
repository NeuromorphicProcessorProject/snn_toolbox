# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 09:44:56 2016

@author: rbodo
"""

import os
import numpy as np
from snntoolbox.core.util import get_sample_activity_from_batch
from snntoolbox.io_utils.plotting import plot_layer_activity


path = '/home/rbodo/.snntoolbox/data/imagenet/vgg16/without_classifier/log/gui/01/activations'
num_samples = 114
batch_size = 2
sample_counter = 0

for batch_idx in range(int(num_samples/batch_size)):
    zipfile = np.load(os.path.join(path, str(batch_idx) + '.npz'))
    activations_batch = zipfile['activations']

    for idx in range(batch_size):
        print("Saving plots of image {}".format(sample_counter))
        activations = get_sample_activity_from_batch(activations_batch, idx)
        for i in range(len(activations)):
            label = activations[i][1]
            newpath = os.path.join(path, label)
            if not os.path.exists(newpath):
                os.makedirs(newpath)
            plot_layer_activity(activations[i], str(sample_counter) +
                                'Activations', newpath)
        sample_counter += 1
