# -*- coding: utf-8 -*-
"""
Created on Thu May 19 16:37:29 2016

@author: rbodo
"""

# For compatibility with python2
from __future__ import print_function, unicode_literals
from __future__ import division, absolute_import
from future import standard_library

import os
import numpy as np
from snntoolbox import echo
from snntoolbox.io_utils.plotting import plot_param_sweep
from snntoolbox.io_utils.load import load_model, get_reshaped_dataset
from snntoolbox.core.SNN import SNN
from snntoolbox.core.util import print_description
from snntoolbox.config import settings

standard_library.install_aliases()


def test_full(queue, params=[settings['v_thresh']], param_name='v_thresh',
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
        - the network architecture (convolutional and fully-connected networks)
        - the dataset (e.g. MNIST or CIFAR10)
        - the spiking simulator to use (currently brian, nest, or Neuron)

    Perform simulations of a spiking network, while optionally sweeping over a
    specified hyper-parameter range. If the keyword arguments are not given,
    the method performs a single run over the specified number of test samples,
    using the updated default parameters.

    Parameters
    ----------

    params : ndarray, optional
        Contains the parameter values for which the simulation will be
        repeated.
    param_name : string, optional
        Label indicating the parameter to sweep, e.g. ``'v_thresh'``.
        Must be identical to the parameter's label in ``globalparams``.
    param_logscale : boolean, optional
        If ``True``, plot test accuracy vs ``params`` in log scale.
        Defaults to ``False``.

    Returns
    -------

    results : list
        List of the accuracies obtained after simulating with each parameter
        value in param_range.

    """

    # Load modified dataset if it has already been stored during previous run,
    # otherwise load it from scratch and perform necessary adaptations (e.g.
    # reducing dataset size for debugging or reshaping according to network
    # architecture). Then save it to disk.
    datadir_base = os.path.expanduser(os.path.join('~', '.snntoolbox'))
    datadir = os.path.join(datadir_base, 'datasets', settings['dataset'],
                           settings['architecture'])
    if not os.path.exists(datadir):
        os.makedirs(datadir)
    samples = os.path.join(datadir, settings['dataset'] + '.npy')
    if os.path.isfile(samples):
        (X_train, Y_train, X_test, Y_test) = tuple(np.load(samples))
    else:
        (X_train, Y_train, X_test, Y_test) = get_reshaped_dataset()
        # Decrease dataset for debugging
        if settings['debug']:
            from random import randint
            ind = randint(0, len(X_test) - settings['batch_size'] - 1)
            X_test = X_test[ind:ind + settings['batch_size']]
            Y_test = Y_test[ind:ind + settings['batch_size']]
        np.save(samples, np.array([X_train, Y_train, X_test, Y_test]))

    # Load model structure and weights.
    model = load_model(settings['filename'])

    # Extract architecture and weights from model.
    snn = SNN(model)
    if settings['verbose'] > 0:
        print_description(snn)

    if not settings['sim_only']:

        # Normalize model
        if settings['normalize']:
            # Evaluate ANN before normalization to ensure it doesn't affect
            # accuracy
            if settings['evaluateANN']:
                echo('\n')
                echo("Before weight normalization:\n")
                snn.evaluate_ann(X_test, Y_test)

            # For memory reasons, reduce number of samples to use during
            # normalization.
            n = 10
            snn.normalize_weights(X_train[:n*settings['batch_size']])

        # (Re-) evaluate ANN
        if settings['evaluateANN']:
            snn.evaluate_ann(X_test, Y_test)

        # Write model to disk
        snn.save(filename=settings['filename_snn'])

        # Compile spiking network from ANN
        snn.build()

        snn.export_to_sim(settings['path'], settings['filename_snn'])

    # Simulate spiking network
    results = []
    for p in params:
        assert param_name in settings, "Unkown parameter"
        settings[param_name] = p

        if len(params) > 1 and settings['verbose'] > 0:
            echo("Current value of parameter to sweep: {} = {:.2f}\n".format(
                                                               param_name, p))
        total_acc = snn.run(X_test, Y_test)

        results.append(total_acc)

    snn.end_sim()

    # Plot and return results of parameter sweep
    n = len(X_test) if settings['simulator'] == 'INI' \
        else settings['num_to_test']
    plot_param_sweep(results, n, params, param_name, param_logscale)

    queue.put(results)
    return results
