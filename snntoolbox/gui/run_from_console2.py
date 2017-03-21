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
            'normalize': False,
            'convert': True,
            'simulate': True,
            'simulator': 'INI',
            'duration': 100,
            'dt': 1,
            'batch_size': 10,
            'verbose': 1,
            'num_to_test': 30,
            'runlabel': '23',
            'percentile': 100,
            'reset': 'Reset by subtraction',  # 'Reset to zero',
            'softmax_to_relu': False,
            'reset_between_frames': True,
            'poisson_input': True,
            'dvs_input': False,
            'num_dvs_events_per_sample': 2000,
            'num_poisson_events_per_sample': 2000,
            'subsample_facs': (239 / 63, 179 / 63),
            'input_rate': 1000,
            'label_dict': {'paper': '0', 'scissors': '1', 'rock': '2',
                           'background': '3'},
            'log_vars': {'activations_n_b_l', 'spiketrains_n_b_l_t',
                         'spikecounts_n_b_l_t', 'input_b_l_t', 'mem_n_b_l_t'},
            'plot_vars': {'activations', 'spiketrains', 'spikecounts',
                          'spikerates', 'input_image', 'error_t',
                          'confusion_matrix', 'correlation'}}

update_setup(settings)

test_full()
