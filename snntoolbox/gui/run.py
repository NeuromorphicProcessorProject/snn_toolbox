# -*- coding: utf-8 -*-

"""
Script to use the toolbox from console instead of GUI.

Created on Mon Mar  7 15:30:28 2016
@author: rbodo
"""

from snntoolbox.config import update_setup
from snntoolbox.core.pipeline import test_full

settings = {'path_wd': '/home/rbodo/.snntoolbox/data/cifar10/xu',
            'dataset_path': '/home/rbodo/.snntoolbox/Datasets/cifar10/mean_subtracted',
            'dataset_format': 'npz',
            'filename_ann': '79.83',
            'model_lib': 'keras',
            'evaluateANN': False,
            'normalize': False,
            'duration': 300,
            'batch_size': 20,
            'num_to_test': 10000,
            'runlabel': '01',
            'percentile': 99,
            'softmax_to_relu': False,
            'log_vars': {'operations_b_t'},
            'plot_vars': set({}),#'activations', 'spikerates', 'input_image',
                          # 'confusion_matrix', 'correlation', 'operations'}
            }

update_setup(settings)

test_full()
