# -*- coding: utf-8 -*-
"""
Usecase:
    1. Specify parameters
    2. Load dataset
    3. Call ``run_SNN``. This will

        - load your already converted SNN
        - run it on a spiking simulator
        - Plot spikerates, spiketrains and membrane voltage.

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

from snntoolbox.config import update_setup
from snntoolbox.io.load import get_reshaped_dataset
from snntoolbox.core.simulation import run_SNN

standard_library.install_aliases()


if __name__ == '__main__':

    # Parameters
    global_params = {'dataset': 'mnist',
                     'architecture': 'mlp',
                     'path': '../data/',
                     'filename': '98.29'}
    cell_params = {'v_thresh': 1.0,
                   'v_reset': 0.0}
    sim_params = {'duration': 1000.0,
                  'dt': 10,
                  'num_to_test': 2}

    # Check that parameter choices are valid. Parameters that were not
    # specified above are filled in from the default parameters.
    update_setup(global_params, cell_params, sim_params)

    # Load dataset, reshaped according to network architecture
    (X_train, Y_train, X_test, Y_test) = get_reshaped_dataset()

    # Simulate spiking network
    total_acc, spiketrains, vmem = run_SNN(X_test, Y_test)
