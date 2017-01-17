# -*- coding: utf-8 -*-

"""
Script to use the toolbox from console instead of GUI.

Created on Mon Mar  7 15:30:28 2016
@author: rbodo
"""

from snntoolbox.config import update_setup
from snntoolbox.core.pipeline import test_full


settings = {'path_wd': '/home/rbodo/.snntoolbox/data/imagenet/inception_averagepool',
            'dataset_path': '/home/rbodo/.snntoolbox/Datasets/imagenet/GoogLeNet',
            'dataset_format': 'npz',
            'filename_ann': '69.70_89.38',
            'model_lib': 'lasagne',
            'evaluateANN': False,
            'normalize': True,
            'convert': True,
            'simulate': True,
            'simulator': 'INI',
            'duration': 550,
            'batch_size': 20,
            'verbose': 1,
            'num_to_test': 10000,
            'runlabel': 'clamp2',
            'percentile': 99.999,
            'normalization_schedule': False,
            }

update_setup(settings)

test_full()