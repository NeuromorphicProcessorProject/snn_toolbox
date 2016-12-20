# -*- coding: utf-8 -*-

"""
This script uses a reduced version of inception-v3, where we cut off all layers
after the first inception module. The data set is not imagenet but comma's
driving data set. The objective is to see whether the transient response of
neurons during simulation can be reduced by keeping the state of the membrane
potential between frames. For this, turn off the reset in ``INI_target_sim.py``.
To prevent the simulator from crashing due to the unexpected output format
(4D-Pool layer instead of 2D-FC layer), need to reshape the output spikes, e.g.:
output += np.argmax(np.reshape(out_spikes.astype('int32'),
                               (out_spikes.shape[0], -1)), axis=1)
For the ``settings['runlabel']`` parameter, choose among 'without_reset',
'with_reset', 'reset_first'.
The output can then be evaluated with the ipython notebook
``tests/other/comma-driving.ipy``

Created on Mon Mar  7 15:30:28 2016
@author: rbodo
"""

from snntoolbox.config import update_setup
from snntoolbox.core.pipeline import test_full


settings = {'path_wd': '/home/rbodo/.snntoolbox/data/imagenet/inception_first',
            'dataset_path': '/home/rbodo/.snntoolbox/Datasets/comma-driving',
            'dataset_format': 'npz',
            'filename_ann': 'comma-driving',
            'model_lib': 'lasagne',
            'evaluateANN': False,
            'normalize': True,
            'convert': True,
            'simulate': True,
            'simulator': 'INI',
            'duration': 100,
            'batch_size': 1,
            'verbose': 3,
            'num_to_test': 100,
            'runlabel': 'reset_first',
            'percentile': 99.999,
            'normalization_schedule': False
            }

update_setup(settings)

test_full()
