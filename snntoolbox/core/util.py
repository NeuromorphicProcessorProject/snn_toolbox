# -*- coding: utf-8 -*-
"""
Helper functions to handle parameters and variables of interest during
conversion and simulation of an SNN.

Created on Wed Mar  9 16:18:33 2016

@author: rbodo
"""

# For compatibility with python2
from __future__ import print_function, unicode_literals
from __future__ import division, absolute_import
from future import standard_library

import numpy as np
from snntoolbox.config import settings

standard_library.install_aliases()


def get_range(start=0.0, stop=1.0, num=5, method='linear'):
    """
    Return a range of parameter values.

    Convenience function. For more flexibility, use ``numpy.linspace``,
    ``numpy.logspace``, ``numpy.random.random_sample`` directly.

    Parameters
    ----------

    start: scalar, optional
        The starting value of the sequence
    stop: scalar, optional
        End value of the sequence.
    num: int, optional
        Number of samples to generate. Must be non-negative.
    method: string, optional
        The sequence will be computed on either a linear, logarithmic or random
        grid.

    Returns
    -------

    samples: ndarray
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


def print_description(snn=None, log=True):
    """
    Print a summary of the test run, parameters, and network. If ``log==True``,
    the output is written as ``settings.txt`` file to the folder given by
    ``settings['log_dir_of_current_run']``.

    """

    if log:
        import os
        f = open(os.path.join(settings['log_dir_of_current_run'],
                              'settings.txt'), 'w')
    else:
        import sys
        f = sys.stdout

    print('\n', file=f)
    print("SUMMARY SETUP", file=f)
    print("=============\n", file=f)
    print("PARAMETERS", file=f)
    print("----------\n", file=f)
    print(settings, file=f)
    print('\n', file=f)
    if snn is not None:
        print("NETWORK", file=f)
        print("-------\n", file=f)
        print(snn.get_config(), file=f)
        print('\n', file=f)
    print("END OF SUMMARY", file=f)
    print('\n', file=f)


def spiketrains_to_rates(spiketrains_batch):
    """
    Convert spiketrains to spikerates.

    The output will have the same shape as the input except for the last
    dimension, which is removed by replacing a sequence of spiketimes by a
    single rate value.

    """

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
                    spikerates_batch[i][0][ii, jj] = (
                        len(np.nonzero(sp[0][ii][jj])[0]) * 1000 /
                        settings['duration'])
        elif len(shape) == 4:
            for ii in range(len(sp[0])):
                for jj in range(len(sp[0][ii])):
                    for kk in range(len(sp[0][ii, jj])):
                        for ll in range(len(sp[0][ii, jj, kk])):
                            spikerates_batch[i][0][ii, jj, kk, ll] = (
                                len(np.nonzero(sp[0][ii, jj, kk, ll])[0]) *
                                1000 / settings['duration'])

    return spikerates_batch


def get_sample_activity_from_batch(activity_batch, i=0):
    """
    Returns layer activity for sample ``i`` of an ``activity_batch``.

    """

    return [(layer_act[0][i], layer_act[1]) for layer_act in activity_batch]


def norm_parameters(parameters, activations, prev_scale_fac):
    """
    Normalize parameters

    Determine the maximum activation of a layer and apply this factor to
    normalize the parameters.

    Parameters
    ----------

    parameters: array
        The parameters of a layer (both weights and biases).
    activations: array
        The activations of cells in a specific layer. Has the same shape as the
        layer.

    Returns
    -------

    parameters_norm: array
        The parameters of a layer, divided by ``scale_fac``.
    scale_fac: float
        Maximum (or percentile) of activations or parameters of this layer.
        Parameters of the respective layer are scaled by this value.

    """

    scale_fac = np.percentile(activations, settings['percentile'])
    print("Maximum value: {:.2f}.".format(scale_fac))
    return [parameters[0] * prev_scale_fac / scale_fac,
            parameters[1] / scale_fac], scale_fac


def get_activations_layer(get_activ, X_train):
    """
    Get activations of a specific layer.

    Parameters
    ----------

    get_activ: Theano function
        A Theano function computing the activations of a layer.

    X_train: float32 array
        The samples to compute activations for. With data of the form
        (channels, num_rows, num_cols), X_train has dimension
        (batch_size, channels*num_rows*num_cols) for a multi-layer perceptron,
        and (batch_size, channels, num_rows, num_cols) for a convolutional net.

    Returns
    -------

    activations: array
        The activations of cells in a specific layer. Has the same shape as the
        layer.

    """

    shape = list(get_activ(X_train[:settings['batch_size']]).shape)
    shape[0] = X_train.shape[0]
    activations = np.empty(shape)
    num_batches = int(np.ceil(X_train.shape[0] / settings['batch_size']))
    for batch_idx in range(num_batches):
        # Determine batch indices.
        max_idx = min((batch_idx + 1) * settings['batch_size'],
                      X_train.shape[0])
        batch_idxs = range(batch_idx * settings['batch_size'], max_idx)
        batch = X_train[batch_idxs, :]
        if len(batch_idxs) < settings['batch_size']:
            batch.resize(X_train[:settings['batch_size']].shape)
            activations[batch_idxs] = get_activ(batch)[:len(batch_idxs)]
        else:
            activations[batch_idxs] = get_activ(batch)
    return activations


def get_activations_batch(ann, X_batch):
    """
    Compute layer activations of an ANN.

    Parameters
    ----------

    ann: SNN
        An instance of the SNN class. Contains the ``get_activ`` function that
        allow computation of layer activations of the original input model
        (hence the name 'ann'). Needed in activation and correlation plots.

    X_batch: float32 array
        The input samples to use for determining the layer activations. With
        data of the form (channels, num_rows, num_cols), X has dimension
        (batch_size, channels*num_rows*num_cols) for a multi-layer perceptron,
        and (batch_size, channels, num_rows, num_cols) for a convolutional net.

    Returns
    -------

    activations_batch: list of tuples ``(activations, label)``
        Each entry represents a layer in the ANN for which an activation can be
        calculated (e.g. ``Dense``, ``Convolution2D``).
        ``activations`` containing the activations of a layer. It has the same
        shape as the original layer, e.g.
        (batch_size, n_features, n_rows, n_cols) for a convolution layer.
        ``label`` is a string specifying the layer type, e.g. ``'Dense'``.

    """

    activations_batch = []
    # Loop through all layers, looking for activation layers
    for idx in range(len(ann.layers)):
        if 'get_activ' not in ann.layers[idx].keys():
            continue
        i = idx if 'Pooling' in ann.layers[idx]['label'] else idx-1
        activations_batch.append((ann.layers[idx]['get_activ'](X_batch),
                                  ann.layers[i]['label']))
    return activations_batch


def wilson_score(p, n):
    """
    Confidence interval of a binomial distribution.

    See https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval.

    Parameters
    ----------

    p: float
        The proportion of successes in ``n`` experiments.
    n: int
        The number of Bernoulli-trials (sample size).

    Returns
    -------

    The confidence interval.

    """

    if n == 0:
        return 0

    # Quantile z of a standard normal distribution, for the error quantile a:
    z = 1.96  # 1.44 for a == 85%, 1.96 for a == 95%
    return (z*np.sqrt((p*(1-p) + z*z/(4*n))/n)) / (1 + z*z/n)


def extract_label(label):
    """
    Get the layer number, name and shape from a string.

    Parameters
    ----------

    label: string
        Specifies both the layer type, index and shape, e.g.
        ``'03Convolution2D_3x32x32'``.

    Returns
    -------

    layer_num: int
        The index of the layer in the network.

    name: string
        The type of the layer.

    shape: tuple
        The shape of the layer
    """

    l = label.split('_')
    layer_num = None
    for i in range(max(4, len(l) - 2)):
        if l[0][:i].isnumeric():
            layer_num = int(l[0][:i])
    name = ''.join(s for s in l[0] if not s.isdigit())
    if name[-1] == 'D':
        name = name[:-1]
    if len(l) > 1:
        shape = tuple([int(s) for s in l[-1].split('x')])
    else:
        shape = ()
    return (layer_num, name, shape)
