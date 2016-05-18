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

import matplotlib.pyplot as plt
import numpy as np

from snntoolbox import echo
from snntoolbox.config import globalparams, cellparams, simparams

standard_library.install_aliases()


#class SNN():
#    def __init__(self):


def test_full(params=[cellparams['v_thresh']],
              param_name='v_thresh',
              param_logscale=False):
    """
    Convert an ANN to a spiking neural network and simulate it.

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

    import os
    import snntoolbox
    from snntoolbox.io.load import load_model, get_reshaped_dataset, ANN
    from snntoolbox.core.conversion import convert_to_SNN
    from snntoolbox.core.simulation import run_SNN

    # Load modified dataset if it has already been stored during previous run,
    # otherwise load it from scratch and perform necessary adaptations (e.g.
    # reducing dataset size for debugging or reshaping according to network
    # architecture). Then save it to disk.
    datadir_base = os.path.expanduser(os.path.join('~', '.snntoolbox'))
    datadir = os.path.join(datadir_base, 'datasets', globalparams['dataset'],
                           globalparams['architecture'])
    if not os.path.exists(datadir):
        os.makedirs(datadir)
    samples = os.path.join(datadir, globalparams['dataset'] + '.npy')
    if os.path.isfile(samples):
        (X_train, Y_train, X_test, Y_test) = tuple(np.load(samples))
    else:
        (X_train, Y_train, X_test, Y_test) = get_reshaped_dataset()
        # Decrease dataset for debugging
        if globalparams['debug']:
            from random import randint
            ind = randint(0, len(X_test) - globalparams['batch_size'] - 1)
            X_test = X_test[ind:ind + globalparams['batch_size'], :]
            Y_test = Y_test[ind:ind + globalparams['batch_size'], :]
        np.save(samples, np.array([X_train, Y_train, X_test, Y_test]))

    # For debugging, use artificial input image
#    f = np.array(np.diag(np.arange(28)), ndmin=3)
#    f[0, 0, :] = 28
#    f[0, :, 0] = 28
#    X_test = np.array([f], ndmin=4)
#    X_test = X_test.astype('float32')
#    X_test /= np.max(X_test)
#    X_train = X_test

    snn = {}

    if globalparams['verbose'] > 0 or not globalparams['sim_only']:
        # Load model structure and weights.
        model = load_model('ann_' + globalparams['filename'])
        print_description(ANN(model['model']))

#    if globalparams['verbose'] > 1:
#        # Display kernels
#        kernels = get_weights(ann)
#        plot_activations(kernels)

    if not globalparams['sim_only']:
        # Normalize ANN
        if globalparams['normalize']:
            from snntoolbox.core.normalization import normalize_weights
            # Evaluate ANN before normalization to ensure it doesn't affect
            # accuracy
            if globalparams['evaluateANN']:
                score = evaluate(model, X_test, Y_test, **{'verbose': 1})
                echo('\n')
                echo("Before weight normalization:\n")
                echo("Test score: {:.2f}\n".format(score[0]))
                echo("Test accuracy: {:.2%}\n".format(score[1]))

            model = normalize_weights(model, X_train,
                                      'ann_' + globalparams['filename'])

        # (Re-) evaluate ANN
        if globalparams['evaluateANN']:
            score = evaluate(model, X_test, Y_test, **{'verbose': 1})
            echo("Test score: {:.2f}\n".format(score[0]))
            echo("Test accuracy: {:.2%}\n".format(score[1]))

        # Extract architecture and weights from model.
        ann = ANN(model['model'])

        # Make sure network has been trained with zero biases
#        if min(ann.biases) != 0 or max(ann.biases) != 0:
#            echo("WARNING: Biases of first layer do not equal zero. " +
#                 "This might cause accuracy loss. " +
#                 "See Diehl et al, 2015, Fast-Classifying...\n")

        # Compile spiking network from ANN
        snn = convert_to_SNN(ann)

    # Simulate spiking network
    results = []
    for p in params:
        if param_name in cellparams:
            cellparams[param_name] = p
        elif param_name in simparams:
            simparams[param_name] = p

        if len(params) > 1 and globalparams['verbose'] > 0:
            echo("Current value of parameter to sweep: {} = {:.2f}\n".format(
                                                               param_name, p))
        total_acc = run_SNN(X_test, Y_test, **snn)

        results.append(total_acc)

    # Compute confidence intervals of the experiments, and plot results.
    if snntoolbox._SIMULATOR == 'INI':
        n = len(X_test)
    else:
        n = simparams['num_to_test']
    ci = [wilson_score(q, n) for q in results]
    ax = plt.subplot()
    if param_logscale:
        ax.set_xscale('log', nonposx='clip')
    ax.errorbar(params, results, yerr=ci, fmt='x-')
    ax.set_title('Accuracy vs Hyperparameter')
    ax.set_xlabel(param_name)
    ax.set_ylabel('accuracy')
    fac = 0.9
    if params[0] < 0:
        fac += 0.2
    ax.set_xlim(fac * params[0], 1.1 * params[-1])
    ax.set_ylim(0, 1)

    return results


def wilson_score(p, n):
    """
    Confidence interval of a binomial distribution.

    See
    ``https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval``

    Parameters
    ----------

    p : float
        The proportion of successes in ``n`` experiments.
    n : int
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


def evaluate(ann, X_test, Y_test, **kwargs):
    """
    Evaluate the performance of a network.

    Wrapper for the evaluation functions of specific input neural network
    libraries ``globalparams['model_lib']`` like keras, caffe, torch, etc.

    Needs to be extended further: Supports only keras so far.

    Parameters
    ----------

    ann : network object
        The neural network of ``backend`` type.
    X_test : float32 array
        The input samples to test.
        With data of the form (channels, num_rows, num_cols),
        X_test has dimension (num_samples, channels*num_rows*num_cols)
        for a multi-layer perceptron, and
        (num_samples, channels, num_rows, num_cols) for a convolutional net.
    Y_test : float32 array
        Ground truth of test data. Has dimesion (num_samples, num_classes).

    Returns
    -------

    The output of the ``model_lib`` specific evaluation function, e.g. the
    score of a keras model.

    """

    if globalparams['model_lib'] == 'keras':
        return ann['model'].evaluate(X_test, Y_test, **kwargs)
    elif globalparams['model_lib'] == 'lasagne':
        def val_epoch(X, y, val_fn):
            """Test a lasagne model batchwise on the whole dataset."""
            err = 0
            loss = 0
            batch_size = globalparams['batch_size']
            batches = int(len(X)/batch_size)

            for i in range(batches):
                new_loss, new_err = val_fn(X[i*batch_size: (i+1)*batch_size],
                                           y[i*batch_size: (i+1)*batch_size])
                err += new_err
                loss += new_loss

            err /= batches
            loss /= batches

            return loss, 1 - err  # Convert error into accuracy here.
        return val_epoch(X_test, Y_test, ann['val_fn'])


def print_description(ann=None):
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
    if ann is not None:
        print("NETWORK")
        print("-------\n")
        print(ann.get_config())
        print('\n')
    print("END OF SUMMARY")
    print('\n')


def spiketrains_to_rates(snn, spiketrains_batch):
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


def get_sample_activity_from_batch(activity_batch):
    """ Returns layer activity for the first sample of a batch. """
    return [(layer_act[0][0], layer_act[1]) for layer_act in activity_batch]


def get_activations(X):
    """ Returns layer activations for a single input X """
    activations_batch = get_activations_batch(np.array(X, ndmin=X.ndim+1))
    return get_sample_activity_from_batch(activations_batch)


def get_activations_batch(X_batch):
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
    import os
    import theano
    from snntoolbox.io.load import load_model
    # Load ANN and compute activations in each layer to compare
    # with SNN spikerates.
    filename = 'ann_' + globalparams['filename']
    if os.path.isfile(os.path.join(globalparams['path'],
                                   filename+'_normWeights.h5')):
        filename += '_normWeights'
        print("DebugMessage: Using normed weights for activations")
    model = load_model(filename)['model']

    activations_batch = []
    # Loop through all layers, looking for activation layers
    if globalparams['model_lib'] == 'keras':
        from keras import backend as K
        layers = model.layers
        for idx, layer in enumerate(layers):
            if layer.__class__.__name__ in {'Activation', 'AveragePooling2D',
                                            'MaxPooling2D'}:
                get_activ = theano.function([layers[0].input,
                                             K.learning_phase()],
                                            layer.output,
                                            allow_input_downcast=True,
                                            on_unused_input='ignore')
                activations_batch.append((get_activ(X_batch, 0),
                                          layers[idx-1].get_config()['name']))
    elif globalparams['model_lib'] == 'lasagne':
        import lasagne
        layers = lasagne.layers.get_all_layers(model)
        for idx, layer in enumerate(layers):
            name = layer.__class__.__name__
            if name == 'DenseLayer':
                label = 'Dense'
            elif name in {'Conv2DLayer', 'Conv2DDNNLayer'}:
                label = 'Convolution2D'
            elif name == 'Pool2DLayer':
                label = 'AveragePooling2D'
            elif name == 'MaxPool2DLayer':
                label = 'MaxPooling2D'
            if name in {'Dense', 'Convolution2D', 'AveragePooling2D',
                        'MaxPooling2D'}:
                get_activ = theano.function(
                    [layers[0].input_var],
                    lasagne.layers.get_output(layer, layers[0].input_var),
                    allow_input_downcast=True)
                activations_batch.append((get_activ(X_batch), label))
    return activations_batch


def extract_label(label):
    """
    Get the layer number, name and shape from a string.

    Parameters
    ----------

    label : string
        Specifies both the layer type, index and shape, e.g.
        ``'3Convolution2D_3x32x32'``.

    Returns
    -------

    layer_num : int
        The index of the layer in the network.

    name : string
        The type of the layer.

    shape : tuple
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


# def get_weights(model):
#    """
#    Return weights of an ANN such that they can be plotted using
#    ``plot_activations``.
#
#    Parameters
#    ----------
#
#    model : ANN model
#        The network architecture and weights as ``snntoolbox.io.load.ANN``
#        object.
#
#    Returns
#    -------
#
#    layer_weights : list of tuples ``(weights, label)``
#        Each entry represents a layer in the ANN containing weights
#        (e.g. ``Dense``, ``Convolution2D``).
#
#        ``weights`` is an array of the same dimension as the corresponding
#        layer, containing its weights.
#
#        ``label`` is a string specifying the layer type, e.g. ``'Dense'``.
#
#    """
#
#    layer_weights = []
#    for layer in model.layers:
#        if 'weights' in layer.keys():
#            layer_weights.append((np.array(layer['weights']),
#                                  layer['layer_type']))
#    return layer_weights
