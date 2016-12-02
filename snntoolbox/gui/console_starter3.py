# -*- coding: utf-8 -*-

"""
Script to use the toolbox from console instead of GUI.

Created on Mon Mar  7 15:30:28 2016
@author: rbodo
"""

from snntoolbox.config import update_setup
from snntoolbox.core.pipeline import test_full


settings = {'path_wd': '/home/rbodo/.snntoolbox/data/mnist/cnn/98.51',
            'dataset_path': '/home/rbodo/.snntoolbox/Datasets/mnist/cnn',
            'dataset_format': 'npz',
            'filename_ann': '98.51',
            'model_lib': 'keras',
            'evaluateANN': False,
            'normalize': False,
            'convert': True,
            'simulate': True,
            'simulator': 'INI',
            'duration': 20,
            'batch_size': 10,
            'verbose': 3,
            'num_to_test': 10000,
            'runlabel': 'test',
            'percentile': 99.999,
            'normalization_schedule': True
            }

update_setup(settings)

test_full()
