# -*- coding: utf-8 -*-
"""
A wrapper script to use all aspects of the toolbox, e.g. converting or
simulating only, or performing a random or grid search to determine the optimal
hyperparameters.

This example uses MNIST dataset and a convolutional network.

The performance of the original ANN lies at 99.16%. The converted network
should be around 98.97% accuracy using built-in INI simulator.

Usecase A: Convert only
    1. Set ``convert = True`` and ``simulate = False``
    2. Specify other parameters (working directory, filename, ...)
    3. Update settings: ``update_setup(settings)``
    4. Call ``test_full()``. This will

        - load the dataset,
        - load a pretrained ANN from ``<path>/<filename>``
        - optionally evaluate it (``evaluate = True``),
        - optionally normalize weights (``normalize = True``),
        - convert to spiking,
        - save SNN to disk.

Usecase B: Simulate only
    1. Set ``convert = False`` and ``simulate = True``
    2. Specify other parameters (working directory, simulator to use, ...)
    3. Update settings: ``update_setup(settings)``
    4. Call ``test_full()``. This will

        - load the dataset,
        - load your already converted SNN,
        - run the net on a spiking simulator,
        - plot spikerates, spiketrains, activations, correlations, etc.

    Note: It is assumed that a network has already been converted (e.g. with
    Usecase A). I.e. there should be a folder in ``<path>`` containing the
    converted network, named ``snn_<filename>_<simulator>``.

Usecase C: Parameter sweep
    1. Specify parameters and update settings with ``update_setup(settings)``
    2. Define a parameter range to sweep, e.g. for `v_thresh`, using for
       instance the helper function ``get_range()``
    3. Call ``test_full``. This will

        - load an already converted SNN or perform a conversion as specified in
          settings.
        - run the SNN repeatedly on a spiking simulator while varying the
          hyperparameter
        - plot accuracy vs. hyperparameter

Usecase C is shown in full in the example below.

Created on Mon Mar  7 15:30:28 2016

@author: rbodo
"""

import snntoolbox


def main():

    # Parameters
    settings = {'dataset_path': 'example/dataset/mnist.npy',  # Dataset file
                'filename': '99.16',
                'path': 'example/',  # Working directory
                'evaluateANN': True,  # Test accuracy of input model
                'normalize': True,  # Normalize weights to get better perf.
                'convert': True,  # Convert analog net to spiking
                'simulate': True,  # Simulate converted net
                'verbose': 3,  # Show plots and temporary results
                'v_thresh': 1,  # Threshold potential
                'simulator': 'INI',  # Use built-in simulator
                'duration': 100.0}  # Simulation time

    # Update defaults with parameters specified above:
    snntoolbox.update_setup(settings)

    # Download and save dataset in npy format which the toolbox can load later.
    from snntoolbox.io_utils.datasets.mnist import get_mnist
    get_mnist('example/dataset/')

    # Run network (including loading the model, weight normalization,
    # conversion and simulation).

    # If set True, the converted model is simulated for three different values
    # of v_thresh. Otherwise use parameters as specified above,
    # for a single run.
    do_param_sweep = False
    if do_param_sweep:
        param_name = 'v_thresh'
        params = snntoolbox.get_range(0.1, 1.5, 3, method='linear')
        snntoolbox.test_full(params=params,
                             param_name=param_name,
                             param_logscale=False)
    else:
        snntoolbox.test_full()

if __name__ == '__main__':
    main()
