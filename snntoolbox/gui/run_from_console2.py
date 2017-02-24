# -*- coding: utf-8 -*-

"""
Script to use the toolbox from console instead of GUI.

Created on Mon Mar  7 15:30:28 2016
@author: rbodo
"""

from snntoolbox.config import update_setup
from snntoolbox.core.pipeline import test_full


settings = {'path_wd': '/home/rbodo/.snntoolbox/data/roshambo',
            'dataset_path': '/home/rbodo/.snntoolbox/Datasets/roshambo',
            'dataset_format': 'npz',
            'filename_ann': 'NullHop',
            'model_lib': 'caffe',
            'evaluateANN': False,
            'normalize': True,
            'convert': True,
            'simulate': True,
            'simulator': 'INI',
            'duration': 10,
            'dt': 1,
            'batch_size': 2,
            'verbose': 1,
            'num_to_test': 4,
            'runlabel': '05',
            'percentile': 100,
            'reset': 'Reset by subtraction',
            'softmax_to_relu': False,
            'reset_between_frames': True,
            'poisson_input': False,
            'log_vars': ['activations', 'spiketrains', 'spikecounts'],
            'plot_vars': ['activations', 'spikecounts', 'spikerates',
             'input_image', 'error_t', 'confusion_matrix', 'correlation',
             'hist_spikerates_activations', 'v_mem']
            }

update_setup(settings)

test_full()
