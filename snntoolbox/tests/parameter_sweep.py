# -*- coding: utf-8 -*-
"""
Usecase:
    1. Specify parameters
    2. Define a parameter range to sweep, e.g. for `v_thresh`
    3. Call ``test_full``. This will

        - load an already converted SNN
        - run it repeatedly on a spiking simulator while varying the
          hyperparameter
        - plot accuracy vs. hyperparameter

This example uses MNIST dataset and a convolutional network.

It is assumed that a network has been converted using for instance the script
``convert_only.py``. (There should be a folder in
``<repo_root>/<path>/<dataset>/<architecture>/`` containing the converted
network.)

For a description of ``global_params``, ``cell_params``, and ``sim_params``,
see ``snntoolbox/config.py``.

Created on Wed Feb 17 09:45:22 2016

@author: rbodo
"""

# For compatibility with python2
from __future__ import print_function, unicode_literals
from __future__ import division, absolute_import
from future import standard_library

from snntoolbox.core.util import get_range, test_full
from snntoolbox.config import update_setup

standard_library.install_aliases()


if __name__ == '__main__':

    # Parameters
    global_params = {'dataset': 'mnist',
                     'architecture': 'cnn',
                     'path': '../data/',
                     'filename': '99.06',
                     'sim_only': True}
    cell_params = {'v_reset': 0.0}
    sim_params = {'duration': 100.0,
                  'dt': 5.0,
                  'num_to_test': 2}

    update_setup(global_params=global_params,
                 cell_params=cell_params,
                 sim_params=sim_params)

    # Define parameter values to sweep
    thresholds = get_range(0.4, 1.5, 2, method='linear')

    # Run simulation for each value in the specified parameter range.
    # The method `test_full` combines and generalizes loading, normalization,
    # evaluation, conversion and simulation steps. It also plots accuracy vs
    # hyperparameter.
    (results, spiketrains, vmem) = test_full(thresholds, 'v_thresh')
