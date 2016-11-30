# -*- coding: utf-8 -*-

"""
Script to use the toolbox from console instead of GUI.

Created on Mon Mar  7 15:30:28 2016
@author: rbodo
"""

from snntoolbox.config import update_setup
from snntoolbox.core.pipeline import test_full


# settings = {'path_wd': '/home/rbodo/.snntoolbox/data/imagenet/vgg16_averagepool',
#             'dataset_path': '/home/rbodo/.snntoolbox/Datasets/imagenet/VGG16',
#             'dataset_format': 'npz',
#             'filename_ann': '63.40',
#             'model_lib': 'keras',
#             'evaluateANN': False,
#             'normalize': True,
#             'convert': True,
#             'simulate': True,
#             'simulator': 'INI',
#             'duration': 200,
#             'batch_size': 20,
#             'verbose': 1,
#             'num_to_test': 10000,
#             'runlabel': 'test',
#             'percentile': 99.999,
#             'normalization_schedule': True
#             }

settings = {'path_wd': '/home/rbodo/.snntoolbox/data/imagenet/vgg16/INI',
            'dataset_path': '/home/rbodo/.snntoolbox/Datasets/imagenet/VGG16',
            'dataset_format': 'npz',
            'filename_ann': '70.88',
            'model_lib': 'keras',
            'evaluateANN': False,
            'normalize': True,
            'convert': True,
            'simulate': True,
            'simulator': 'INI',
            'duration': 200,
            'batch_size': 1,
            'verbose': 3,
            'num_to_test': 10000,
            'runlabel': 'test',
            'percentile': 99.999,
            'normalization_schedule': True
            }

update_setup(settings)

test_full()
