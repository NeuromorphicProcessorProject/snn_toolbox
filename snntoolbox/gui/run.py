# -*- coding: utf-8 -*-

"""
Script to use the toolbox from console instead of GUI.

Created on Mon Mar  7 15:30:28 2016
@author: rbodo
"""

from snntoolbox.config import update_setup
from snntoolbox.core.pipeline import test_full

settings = {'path_wd': '/home/rbodo/.snntoolbox/data/cifar10/88.22',
            'dataset_path': '/home/rbodo/.snntoolbox/Datasets/cifar10/binarynet',
            'dataset_format': 'npz',
            'filename_ann': '88.22',
            'model_lib': 'lasagne',
            'normalize': False,
            'evaluateANN': False,
            'duration': 3,
            'batch_size': 10,
            'num_to_test': 10000,
            'runlabel': '01',
            'maxpool_type': 'binary_sigmoid',
            'binarize_weights': True,
            'log_vars': {'operations_b_t'},
            'plot_vars': {'activations', 'spikerates', 'input_image',
                          'confusion_matrix', 'correlation', 'operations'}
            }

update_setup(settings)

test_full()
