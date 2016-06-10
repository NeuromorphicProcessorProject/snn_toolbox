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

from snntoolbox.io_utils.plotting import plot_param_sweep
from snntoolbox.io_utils.load import load_dataset
from snntoolbox.core.model import SNN
from snntoolbox.core.util import print_description
from snntoolbox.config import settings

standard_library.install_aliases()


def test_full(queue=None, params=[settings['v_thresh']], param_name='v_thresh',
              param_logscale=False):
    """
    Convert an snn to a spiking neural network and simulate it.

    Complete pipeline of
        1. loading and testing a pretrained ANN,
        2. normalizing weights
        3. converting it to SNN,
        4. running it on a simulator,
        5. if given a specified hyperparameter range ``params``,
           repeat simulations with modified parameters.

    The testsuit allows specification of
        - the dataset (e.g. MNIST or CIFAR10)
        - the spiking simulator to use (currently Brian, Brian2, Nest, Neuron,
          or INI's simulator.)

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
    (X_train, Y_train, X_test, Y_test) = load_dataset(settings['dataset_path'])

    # _____________________________ LOAD MODEL ______________________________ #
    # Extract architecture and weights from input model.
    snn = SNN(settings['path'], settings['filename'])

    if settings['verbose'] > 0:
        print_description(snn)

    if settings['convert'] and not is_stop(queue):

        # ____________________________ NORMALIZE ____________________________ #
        # Normalize model
        if settings['normalize'] and not is_stop(queue):
            # Evaluate ANN before normalization to ensure it doesn't affect
            # accuracy
            if settings['evaluateANN'] and not is_stop(queue):
                print("Before weight normalization:")
                snn.evaluate_ann(X_test, Y_test)

            # For memory reasons, reduce number of samples to use during
            # normalization.
            frac = 1
            snn.normalize_weights(X_train[:int(len(X_train)/frac)])

        # ____________________________ EVALUATE _____________________________ #
        # (Re-) evaluate ANN
        if settings['evaluateANN'] and not is_stop(queue):
            snn.evaluate_ann(X_test, Y_test)

        # _____________________________ EXPORT ______________________________ #
        # Write model to disk
        snn.save()

        # Compile spiking network from ANN
        if not is_stop(queue):
            snn.build()

        # Export network in a format specific to the simulator with wich it
        # will be tested later.
        if not is_stop(queue):
            snn.export_to_sim()

    # ______________________________ SIMULATE _______________________________ #
    results = []
    if settings['simulate'] and not is_stop(queue):

        if len(params) > 1 and settings['verbose'] > 0:
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
                print("Current value of parameter to sweep: " +
                      "{} = {:.2f}\n".format(param_name, p))
            # Simulate network
            total_acc = snn.run(X_test, Y_test)

            # Write out results
            results.append(total_acc)

        # Clean up
        snn.end_sim()

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
