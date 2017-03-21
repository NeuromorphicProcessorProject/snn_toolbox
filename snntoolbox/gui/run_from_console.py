# -*- coding: utf-8 -*-

"""
Script to use the toolbox from console instead of GUI.

Created on Mon Mar  7 15:30:28 2016
@author: rbodo
"""

from snntoolbox.config import update_setup
from snntoolbox.core.pipeline import test_full

import json

class_idx_path = '/home/rbodo/.snntoolbox/Datasets/imagenet/GoogLeNet/imagenet_class_index.json'
class_idx = json.load(open(class_idx_path, "r"))
classes = [class_idx[str(idx)][0] for idx in range(len(class_idx))]

settings = {'path_wd': '/home/rbodo/.snntoolbox/data/imagenet/inception_lasagne',
            'dataset_path': '/home/rbodo/.snntoolbox/Datasets/imagenet/validation',
            'dataset_format': 'jpg',
            'dataflow_kwargs': str({'target_size': (299, 299),
                                    'classes': classes, 'shuffle': False}),
            'filename_ann': 'inception',
            'model_lib': 'lasagne',
            'evaluateANN': False,
            'normalize': True,
            'convert': True,
            'simulate': True,
            'simulator': 'INI',
            'duration': 600,
            'batch_size': 25,
            'verbose': 1,
            'num_to_test': 50000,
            'runlabel': 'clamp',
            'percentile': 99.999,
            'softmax_to_relu': True,
            'log_vars': [],  # ['spikecounts', 'spiketrains'],  # 'activations'
            'plot_vars': []# '['spikecounts', 'error_t', 'spikerates']
            }

update_setup(settings)

test_full()
