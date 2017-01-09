# -*- coding: utf-8 -*-

"""
Script to use the toolbox from console instead of GUI.

Created on Mon Mar  7 15:30:28 2016
@author: rbodo
"""

from snntoolbox.config import update_setup
from snntoolbox.core.pipeline import test_full


settings = {'path_wd': '/home/rbodo/.snntoolbox/data/imagenet/inception_lasagne',
            'dataset_path': '/home/rbodo/.snntoolbox/Datasets/imagenet/GoogLeNet',
            'dataset_format': 'npz',
            'filename_ann': 'inception',
            'model_lib': 'lasagne',
            'evaluateANN': False,
            'normalize': True,
            'convert': True,
            'simulate': True,
            'simulator': 'INI',
            'duration': 200,
            'batch_size': 1,
            'verbose': 2,
            'num_to_test': 1,
            'runlabel': 'exp_max2',
            'percentile': 100,
            'normalization_schedule': False,
            'maxpool_type': 'exp_max'
            }

update_setup(settings)

test_full()
