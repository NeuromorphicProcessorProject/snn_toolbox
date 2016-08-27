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
        - optionally normalize parameters (``normalize = True``),
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
import os
import numpy as np
import snntoolbox


def main():

    # Parameters
    settings = {'dataset_path': './dataset',  # Dataset file
                'filename_ann': '83.62',
                'path_wd': '.',  # Working directory
                'evaluateANN': True,
                'normalize': True,
                'percentile': 99,
                'batch_size': 100,                
                'overwrite': True,
                'convert': True,
                'simulate': True,
                'verbose': 3,
                'v_thresh': 1,
                'tau_refrac': 0,
                'softmax_clockrate': 300,
                'simulator': 'INI',
                'duration': 100,
                'dt': 1,
                'poisson_input': False,
                'reset': 'Reset by subtraction',
                'input_rate': 1000,
                'normalization_schedule': False,
                'online_normalization': False,
                'payloads': True,          
                'diff_to_max_rate': 200,
                'timestep_fraction': 10,
                'diff_to_min_rate': 100,
                'scaling_factor' : 10000000,
                'maxpool_type': "fir_max"}


    # Update defaults with parameters specified above:
    snntoolbox.update_setup(settings)

    # Download and save dataset in npy format which the toolbox can load later.
    from snntoolbox.io_utils.cifar10_load import get_cifar10
    try: 
        os.makedirs("./dataset")
    except OSError:
        if not os.path.isdir("./dataset"):
            raise  
    get_cifar10(settings['dataset_path'])

    # Run network (including loading the model, parameter normalization,
    # conversion and simulation).

    # If set True, the converted model is simulated for three different values
    # of v_thresh. Otherwise use parameters as specified above,
    # for a single run.
    do_param_sweep = True
    params = [1, 5, 10, 15, 20, 25, 35, 50, 75, 100]
    network_runs = 5
    results = np.zeros((network_runs, len(params)))
    if do_param_sweep:
        for n in range(network_runs):
            param_name = 'duration'
            #params = snntoolbox.get_range(1, 101, 21, method='linear')
            #param_name = 'v_thresh'
            #params = snntoolbox.get_range(0.1, 1.5, 3, method='linear')
            results[n, :] = snntoolbox.test_full(params=params,
                                 param_name=param_name,
                                 param_logscale=False)
            if settings["payloads"]:
                np.savetxt("results_with_payloads", results)
            else:
                np.savetxt("results_without_payloads", results)
                                 
                                 
    else:
        snntoolbox.test_full()

if __name__ == '__main__':
    import pdb
    #pdb.set_trace()
    main()
