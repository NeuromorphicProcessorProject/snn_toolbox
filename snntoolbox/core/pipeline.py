# -*- coding: utf-8 -*-
"""
Wrapper script that combines all tools of SNN Toolbox.

Created on Thu May 19 16:37:29 2016

@author: rbodo
"""

# For compatibility with python2
from __future__ import print_function, unicode_literals
from __future__ import division, absolute_import
from future import standard_library

from importlib import import_module

import os
from snntoolbox.io_utils.plotting import plot_param_sweep
from snntoolbox.io_utils.common import load_dataset
from snntoolbox.core.util import print_description, normalize_parameters, parse
from snntoolbox.config import settings

standard_library.install_aliases()


def test_full(queue=None, params=[settings['v_thresh']], param_name='v_thresh',
              param_logscale=False):
    """Convert an snn to a spiking neural network and simulate it.

    Complete pipeline of
        1. loading and testing a pretrained ANN,
        2. normalizing parameters
        3. converting it to SNN,
        4. running it on a simulator,
        5. if given a specified hyperparameter range ``params``,
           repeat simulations with modified parameters.

    The testsuit allows specification of
        - the dataset (e.g. MNIST or CIFAR10)
        - the spiking simulator to use (currently Brian, Brian2, Nest, Neuron,
          MegaSim or INI's simulator.)

    Perform simulations of a spiking network, while optionally sweeping over a
    specified hyper-parameter range. If the keyword arguments are not given,
    the method performs a single run over the specified number of test samples,
    using the updated default parameters.

    Parameters
    ----------

    queue: Queue, optional
        Results are added to the queue to be displayed in the GUI.
    params: ndarray, optional
        Contains the parameter values for which the simulation will be
        repeated.
    param_name: string, optional
        Label indicating the parameter to sweep, e.g. ``'v_thresh'``.
        Must be identical to the parameter's label in ``globalparams``.
    param_logscale: boolean, optional
        If ``True``, plot test accuracy vs ``params`` in log scale.
        Defaults to ``False``.

    Returns
    -------

    results: list
        List of the accuracies obtained after simulating with each parameter
        value in param_range.
    """

    # ____________________________ LOAD DATASET _____________________________ #
    X_test = load_dataset(settings['dataset_path'], 'X_test.npz')
    Y_test = load_dataset(settings['dataset_path'], 'Y_test.npz')

    # Instantiate an empty spiking network
    target_sim = import_module('snntoolbox.target_simulators.' +
                               settings['simulator'] + '_target_sim')
    spiking_model = target_sim.SNN()  # t=0.1%

    if settings['convert'] and not is_stop(queue):

        # ___________________________ LOAD MODEL ____________________________ #
        # Extract architecture and parameters from input model.
        model_lib = import_module('snntoolbox.model_libs.' +
                                  settings['model_lib'] + '_input_lib')
        input_model = model_lib.load_ann(settings['path'],
                                         settings['filename'])
        if settings['evaluateANN']:
            print('\n' + "Evaluating input model...")
            score = model_lib.evaluate(input_model['val_fn'], X_test, Y_test)
            print('\n' + "Test score: {:.2f}".format(score[0]))
            print("Test accuracy: {:.2%}\n".format(score[1]))

        print("Parsing input model...")
        parsed_model = parse(input_model['model'])  # t=0.5% m=0.6GB
        print_description(parsed_model)

        # ____________________________ NORMALIZE ____________________________ #
        # Normalize model
        if settings['normalize'] and not is_stop(queue):
            # Evaluate ANN before normalization to ensure it doesn't affect
            # accuracy
            if settings['evaluateANN'] and not is_stop(queue):
                print('\n' + "Evaluating parsed model before parameter " +
                      "normalization...")
                score = parsed_model.evaluate(X_test, Y_test)
                print('\n' + "Test score: {:.2f}".format(score[0]))
                print("Test accuracy: {:.2%}\n".format(score[1]))
            # t=9.1 (t=90.8% without sim) m=0.2GB
            normalize_parameters(parsed_model)

        # ____________________________ EVALUATE _____________________________ #
        # (Re-) evaluate ANN
        if settings['evaluateANN'] and not is_stop(queue):
            print('\n' + "Evaluating parsed model...")
            score = parsed_model.evaluate(X_test, Y_test)
            print('\n' + "Test score: {:.2f}".format(score[0]))
            print("Test accuracy: {:.2%}\n".format(score[1]))

        # _____________________________ EXPORT ______________________________ #
        # Write model to disk
        parsed_model.save(os.path.join(
            settings['path'], settings['filename_parsed_model'] + '.h5'))

        # Compile spiking network from ANN
        if not is_stop(queue):
            spiking_model.build(parsed_model)

        # Export network in a format specific to the simulator with which it
        # will be tested later.
        if not is_stop(queue):
            spiking_model.save(settings['path'], settings['filename_snn'])

    # ______________________________ SIMULATE _______________________________ #
    results = []
    if settings['simulate'] and not is_stop(queue):

        if len(params) > 1:
            print("Testing SNN for hyperparameter values {} = ".format(
                param_name))
            print(['{:.2f}'.format(i) for i in params])
            print('\n')

        # Loop over parameter to sweep
        for p in params:
            if is_stop(queue):
                break

            assert param_name in settings, "Unkown parameter"
            settings[param_name] = p

            # Display current parameter value
            if len(params) > 1 and settings['verbose'] > 0:
                print('\n')
                print("Current value of parameter to sweep: " +
                      "{} = {:.2f}\n".format(param_name, p))
            # Simulate network
            total_acc = spiking_model.run(X_test, Y_test)  # t=90.1% m=2.3GB

            # Write out results
            results.append(total_acc)

        # Clean up
        spiking_model.end_sim()

    # _______________________________ OUTPUT _______________________________ #
    # Number of samples used in one run:
    n = len(X_test) if settings['simulator'] == 'INI' \
        else settings['num_to_test']
    # Plot and return results of parameter sweep.
    if results != []:
        plot_param_sweep(results, n, params, param_name, param_logscale)

    # Add results to queue to be displayed in GUI.
    if queue:
        queue.put(results)

    return results


def is_stop(queue):
    if not queue:
        return False
    if queue.empty():
        return False
    elif queue.get_nowait() == 'stop':
        print("Skipped step after user interrupt")
        queue.put('stop')
        return True
