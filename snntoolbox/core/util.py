# -*- coding: utf-8 -*-
"""
Helper functions to run various tests on spiking networks.

Created on Wed Mar  9 16:18:33 2016

@author: rbodo
"""

# For compatibility with python2
from __future__ import print_function, unicode_literals
from __future__ import division, absolute_import
from future import standard_library

import numpy as np
from snntoolbox.config import globalparams, cellparams, simparams

standard_library.install_aliases()


def get_range(start=0.0, stop=1.0, num=5, method='linear'):
    """
    Return a range of parameter values.

    Convenience function. For more flexibility, use ``numpy.linspace``,
    ``numpy.logspace``, ``numpy.random.random_sample`` directly.

    Parameters
    ----------

    start : scalar, optional
        The starting value of the sequence
    stop : scalar, optional
        End value of the sequence.
    num : int, optional
        Number of samples to generate. Must be non-negative.
    method : string, optional
        The sequence will be computed on either a linear, logarithmic or random
        grid.

    Returns
    -------

    samples : ndarray
        There are ``num`` samples in the closed interval [start, stop].
    """

    methods = {'linear', 'log', 'random'}
    assert method in methods, "Specified grid-search method {} not supported.\
        Choose among {}".format(method, methods)
    assert start < stop, "Start must be smaller than stop."
    assert num > 0 and isinstance(num, int), \
        "Number of samples must be unsigned int."
    if method == 'linear':
        return np.linspace(start, stop, num)
    if method == 'log':
        return np.logspace(start, stop, num, endpoint=False)
    if method == 'random':
        return np.random.random_sample(num) * (stop-start) + start


def print_description(snn=None):
    """
    Print a summary of the test run, parameters, and network.

    """

    print('\n')
    print("SUMMARY SETUP")
    print("=============\n")
    print("PARAMETERS")
    print("----------\n")
    print("Global parameters: {}".format(globalparams))
    print('\n')
    print("Cell parameters: {}".format(cellparams))
    print('\n')
    print("Simulation parameters: {}".format(simparams))
    print('\n')
    if snn is not None:
        print("NETWORK")
        print("-------\n")
        print(snn.get_config())
        print('\n')
    print("END OF SUMMARY")
    print('\n')


def spiketrains_to_rates(spiketrains_batch):
    spikerates_batch = []
    for (i, sp) in enumerate(spiketrains_batch):
        shape = sp[0].shape[:-1]  # output_shape of layer
        # Allocate list containing an empty array of shape
        # 'output_shape' for each layer of the network, which will
        # hold the spikerates of a mini-batch.
        spikerates_batch.append((np.empty(shape), sp[1]))
        # Count number of spikes fired in the layer and divide by the
        # simulation time in seconds to get the mean firing rate of each
        # neuron in Hertz.
        if len(shape) == 2:
            for ii in range(len(sp[0])):
                for jj in range(len(sp[0][ii])):
                    spikerates_batch[i][0][ii, jj] = \
                        (len(np.nonzero(sp[0][ii][jj])[0]) * 1000 /
                         simparams['duration'])
        elif len(shape) == 4:
            for ii in range(len(sp[0])):
                for jj in range(len(sp[0][ii])):
                    for kk in range(len(sp[0][ii, jj])):
                        for ll in range(len(sp[0][ii, jj, kk])):
                            spikerates_batch[i][0][ii, jj, kk, ll] = (
                                len(np.nonzero(sp[0][ii, jj, kk, ll])[0]) /
                                simparams['duration'] * 1000)

    return spikerates_batch


def get_sample_activity_from_batch(activity_batch, i=0):
    """ Returns layer activity for the first sample of a batch. """
    return [(layer_act[0][i], layer_act[1]) for layer_act in activity_batch]


def get_activations(X):
    """ Returns layer activations for a single input X """
    activations_batch = get_activations_batch(np.array(X, ndmin=X.ndim+1))
    return get_sample_activity_from_batch(activations_batch)


def get_activations_batch(ann, X_batch):
    """
    Compute layer activations of an ANN.

    Parameters
    ----------

    X_batch : float32 array
        The input samples to use for determining the layer activations.
        With data of the form (channels, num_rows, num_cols), X has dimension
        (1, channels*num_rows*num_cols) for a multi-layer perceptron, and
        (1, channels, num_rows, num_cols) for a convolutional net.

    Returns
    -------

    activations_batch : list of tuples ``(activations, label)``
        Each entry represents a layer in the ANN for which an activation can be
        calculated (e.g. ``Dense``, ``Convolution2D``).

        ``activations`` is an array of the same dimension as the corresponding
        layer, containing the activations of Dense or Convolution layers.

        ``label`` is a string specifying the layer type, e.g. ``'Dense'``.

    """

    activations_batch = []
    # Loop through all layers, looking for activation layers
    for idx in range(len(ann.layers)):
        # Use normalized model if possible.
        if 'get_activ_norm' in ann.layers[idx].keys():
            get_activ = ann.layers[idx]['get_activ_norm']
        elif 'get_activ' in ann.layers[idx].keys():
            get_activ = ann.layers[idx]['get_activ']
        else:
            continue
        i = idx if 'Pooling' in ann.layers[idx]['label'] else idx-1
        activations_batch.append((get_activ(X_batch), ann.layers[i]['label']))
    return activations_batch
