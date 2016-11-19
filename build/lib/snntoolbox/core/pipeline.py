# -*- coding: utf-8 -*-
"""
Wrapper script that combines all tools of SNN Toolbox.

Created on Thu May 19 16:37:29 2016

@author: rbodo
"""

# For compatibility with python2
from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import os
from importlib import import_module

import numpy as np
from future import standard_library
from snntoolbox.config import settings
from snntoolbox.core.util import evaluate_keras, get_dataset
from snntoolbox.core.util import print_description, normalize_parameters, parse
from snntoolbox.io_utils.plotting import plot_param_sweep

standard_library.install_aliases()


def test_full(queue=None, params=None, param_name='v_thresh',
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

    queue: Optional[Queue.Queue]
        Results are added to the queue to be displayed in the GUI.
    params: Optional[list[int]]
        Contains the parameter values for which the simulation will be
        repeated.
    param_name: string
        Label indicating the parameter to sweep, e.g. ``'v_thresh'``.
        Must be identical to the parameter's label in ``globalparams``.
    param_logscale: bool
        If ``True``, plot test accuracy vs ``params`` in log scale.
        Defaults to ``False``.

    Returns
    -------

    results: list
        List of the accuracies obtained after simulating with each parameter
        value in param_range.
    """

    if params is None:
        params = [settings[param_name]]

    # ____________________________ LOAD DATASET _____________________________ #
    evalset, normset, testset = get_dataset(settings)

    # Instantiate an empty spiking network
    input_model = {}
    target_sim = import_module('snntoolbox.target_simulators.' +
                               settings['simulator'] + '_target_sim')
    spiking_model = target_sim.SNN()

    if (settings['evaluateANN'] or settings['convert']) and not is_stop(queue):
        # ___________________________ LOAD MODEL ____________________________ #
        # Extract architecture and parameters from input model.
        model_lib = import_module('snntoolbox.model_libs.' +
                                  settings['model_lib'] + '_input_lib')
        input_model = model_lib.load_ann(settings['path_wd'],
                                         settings['filename_ann'])
        # ____________________________ EVALUATE _____________________________ #
        # Evaluate ANN
        if settings['evaluateANN']:
            print('\n' + "Evaluating input model...")
            score = model_lib.evaluate(input_model['val_fn'], **evalset)
            spiking_model.ANN_err = 1-score[1]

    if settings['convert'] and not is_stop(queue):
        # _____________________________ PARSE ________________________________ #
        # Parse ANN to a Keras model with only layers that can be converted.
        print("Parsing input model...")
        parsed_model = parse(input_model['model'])
        print_description(parsed_model)

        # ____________________________ EVALUATE _____________________________ #
        # (Re-) evaluate ANN
        if settings['evaluateANN'] and not is_stop(queue):
            print('\n' + "Evaluating parsed model...")
            evaluate_keras(parsed_model, **evalset)

        save_activations = False
        if save_activations:
            from snntoolbox.core.util import get_activations_batch
            print("Saving activations")
            path = os.path.join(settings['log_dir_of_current_run'],
                                'activations')
            if not os.path.isdir(path):
                os.makedirs(path)
            num_batches = int(np.floor(
                settings['num_to_test'] / settings['batch_size']))
            for batch_idx in range(num_batches):
                batch_idxs = range(settings['batch_size'] * batch_idx,
                                   settings['batch_size'] * (batch_idx + 1))
                x_batch = evalset['x_test'][batch_idxs, :]
                activations_batch = get_activations_batch(parsed_model,
                                                          x_batch)
                np.savez_compressed(os.path.join(path, str(batch_idx)),
                                    activations=activations_batch,
                                    spiketrains=None)

        # ____________________________ NORMALIZE ____________________________ #
        # Normalize model
        if settings['normalize'] and not is_stop(queue):
            normalize_parameters(parsed_model, **normset)

        # ____________________________ EVALUATE _____________________________ #
        # (Re-) evaluate ANN
        if settings['evaluateANN'] and not is_stop(queue):
            print('\n' + "Evaluating parsed model...")
            evaluate_keras(parsed_model, **evalset)

        # __________________________ SAVE PARSED _____________________________ #
        # Write parsed model to disk
        parsed_model.save(os.path.join(
            settings['path_wd'], settings['filename_parsed_model'] + '.h5'))

        # ____________________________ BUILD SNN _____________________________ #
        # Compile spiking network from ANN
        if not is_stop(queue):
            spiking_model.build(parsed_model)

        # ____________________________ SAVE SNN ______________________________ #
        # Export network in a format specific to the simulator with which it
        # will be tested later.
        if not is_stop(queue):
            spiking_model.save(settings['path_wd'], settings['filename_snn'])

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
            total_acc = spiking_model.run(**testset)

            # Write out results
            results.append(total_acc)

        # Clean up
        spiking_model.end_sim()

    # _______________________________ OUTPUT _______________________________ #
    # Number of samples used in one run:
    n = settings['num_to_test']
    # Plot and return results of parameter sweep.
    plot_param_sweep(results, n, params, param_name, param_logscale)

    # Add results to queue to be displayed in GUI.
    if queue:
        queue.put(results)

    return results


def is_stop(queue):
    """Determine if the user pressed 'stop' in the GUI.

    :param queue: Event queue.
    :return: ``True`` if user pressed 'stop' in GUI, ``False`` otherwise.
    """

    if not queue:
        return False
    if queue.empty():
        return False
    elif queue.get_nowait() == 'stop':
        print("Skipped step after user interrupt")
        queue.put('stop')
        return True
