# -*- coding: utf-8 -*-

"""
Script to use the toolbox from console instead of GUI.

Created on Mon Mar  7 15:30:28 2016
@author: rbodo
"""

from snntoolbox.config import update_setup
from snntoolbox.core.pipeline import test_full

settings = {'path_wd': '/home/rbodo/.snntoolbox/data/mnist/cnn/binarynet',
            'dataset_path': '/home/rbodo/.snntoolbox/Datasets/mnist/cnn',
            'dataset_format': 'npz',
            'filename_ann': '98.93',
            'model_lib': 'keras',
            'normalize': False,
            'evaluateANN': True,
            'duration': 30,
            'batch_size': 100,
            'num_to_test': 10000,
            'runlabel': '01',
            'reset': 'Reset to zero',
            'softmax_to_relu': False,
            'maxpool_type': 'binary_sigmoid',
            'binarize_weights': True,
            'log_vars': {'operations_b_t'},
            'plot_vars': set({})
            }

update_setup(settings)

test_full()
