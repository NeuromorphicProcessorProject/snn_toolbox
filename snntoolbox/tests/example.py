# -*- coding: utf-8 -*-
"""
A wrapper script to use all aspects of the toolbox, e.g. converting or
simulating only, or performing a random or grid search to determine the optimal
hyperparameters.

Created on Mon Mar  7 15:30:28 2016

@author: rbodo
"""

from snntoolbox.config import update_setup
from snntoolbox.core.util import test_full

if __name__ == '__main__':

    # Parameters
    globalparams = {'dataset': 'mnist',
                    'architecture': 'cnn',
                    'filename': '99.06',
                    'path': 'example/mnist/cnn/99.06/INI/',
                    'evaluateANN': True,
                    'debug': False,
                    'batch_size': 100,
                    'normalize': True,
                    'sim_only': False,
                    'verbose': 3}
    cellparams = {'v_thresh': 1}
    simparams = {'duration': 100.0,
                 'max_f': 1000,
                 'num_to_test': 2}

    update_setup(globalparams, cellparams, simparams)

    # Run network (including loading the model, weight normalization,
    # conversion and simulation)
    test_full()
#    from snntoolbox.core.util import get_range
#    param_name = 'v_thresh'
#    params = get_range(0.1, 1.5, 2, method='linear')
#    print("Testing SNN for hyperparameter values {} = ".format(param_name))
#    print(['{:.2f}'.format(i) for i in params])
#    print('\n')
#    test_full(params=params, param_name=param_name, param_logscale=False)
