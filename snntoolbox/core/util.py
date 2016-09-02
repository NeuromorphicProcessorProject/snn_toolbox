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

import os
import theano
import numpy as np
from keras import backend as K
from importlib import import_module
from snntoolbox.config import settings

standard_library.install_aliases()


def parse(input_model):
    """Create a Keras model suitable for conversion to SNN.

    This parsing function takes as input an arbitrary neural network and builds
    a Keras model from it with the same functionality and performance.
    The resulting model contains all essential information about the network,
    independently of the model library in which the original network was built
    (e.g. Caffe). This makes the SNN toolbox stable against changes in input
    formats. Another advantage is extensibility: In order to add a new input
    language to the toolbox (e.g. Lasagne), a developer only needs to add a
    single module to ``model_libs`` package, implementing a number of methods
    (see the respective functions in 'keras_input_lib.py' for more details.)

    Parameters
    ----------

    input_model: Analog Neural Network
        A pretrained neural network model.

    Returns
    -------

    parsed_model: Keras model
        A Keras model functionally equivalent to ``input_model``.
    """

    import keras

    model_lib = import_module('snntoolbox.model_libs.' +
                              settings['model_lib'] + '_input_lib')
    # Parse input model to our common format, extracting all necessary
    # information about layers.
    layers = model_lib.extract(input_model)
    # Create new Keras model
    parsed_model = keras.models.Sequential()
    for layer in layers:
        # Replace 'parameters' key with Keras key 'weights'
        if 'parameters' in layer:
            layer['weights'] = layer.pop('parameters')
        # Remove keys that are not understood by Keras layer constructor
        layer_type = layer.pop('layer_type')
        filter_flip = layer.pop('filter_flip', None)
        # Add layer
        parsed_layer = getattr(keras.layers, layer_type)
        parsed_model.add(parsed_layer(**layer))
        if 'filter_flip' in layer:
            parsed_layer.filter_flip = filter_flip
    # Optimizer and loss should not matter at this stage, but it would be
    # cleaner to set them to the actial values of the input model.
    parsed_model.compile('sgd', 'categorical_crossentropy',
                         metrics=['accuracy'])
    return parsed_model


def evaluate_keras(model, X_test=None, Y_test=None, dataflow=None):
    """Evaluate parsed Keras model.

    Can use either numpy arrays ``X_test, Y_test`` containing the test samples,
    or generate them with a dataflow
    (``Keras.ImageDataGenerator.flow_from_directory`` object).
    """

    if X_test is not None:
        score = model.evaluate(X_test, Y_test)
    elif dataflow:
        batch_size = dataflow.batch_size
        dataflow.batch_size = settings['num_to_test']
        score = model.evaluate_generator(dataflow, settings['num_to_test'])
        dataflow.batch_size = batch_size
    print('\n' + "Test loss: {:.2f}".format(score[0]))
    print("Test accuracy: {:.2%}\n".format(score[1]))
    return score


def get_range(start=0.0, stop=1.0, num=5, method='linear'):
    """Return a range of parameter values.

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
    """Convert spiketrains to spikerates.

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


def get_sample_activity_from_batch(activity_batch, idx=0):
    """Return layer activity for sample ``idx`` of an ``activity_batch``.
    """

    return [(layer_act[0][idx], layer_act[1]) for layer_act in activity_batch]


def normalize_parameters(model, dataflow=None):
    """Normalize the parameters of a network.

    The parameters of each layer are normalized with respect to the maximum
    activation, or the ``n``-th percentile of activations.

    Generates plots of the activity- and weight-distribution before and after
    normalization. Note that plotting the activity-distribution can be very
    time- and memory-consuming for larger networks.
    """

    from snntoolbox.io_utils.plotting import plot_hist

    if dataflow is None:
        from snntoolbox.io_utils.common import load_dataset
        print("Loading normalization data set from '.npz' file.\n")
        X_norm = load_dataset(settings['dataset_path'], 'X_norm.npz')  # t=0.2%
    else:
        print("Loading normalization data set from ImageDataGenerator.\n")
        X_norm, Y = dataflow.next()

    print("Using {} samples for normalization.".format(len(X_norm)))

#        import numpy as np
#        sizes = [len(X_norm) * np.array(layer['output_shape'][1:]).prod() *
#                 32 / (8 * 1e9) for idx, layer in enumerate(self.layers)
#                 if idx != 0 and 'parameters' in self.layers[idx-1]]
#        size_str = ['{:.2f}'.format(s) for s in sizes]
#        print("INFO: Need {} GB for layer activations.\n".format(size_str) +
#              "May have to reduce size of data set used for normalization.\n")

    print("Normalizing parameters:\n")
    newpath = os.path.join(settings['log_dir_of_current_run'], 'normalization')
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    # Loop through all layers, looking for layers with parameters
    scale_fac_prev_layer = 1
    for idx, layer in enumerate(model.layers):
        # Skip layer if not preceeded by a layer with parameters
        if layer.get_weights() == []:
            continue
        if settings['verbose'] > 1:
            print("Calculating activation of layer {} with shape {}...".format(
                  layer.name, layer.output_shape))
        parameters = layer.get_weights()
        # Undo previous scaling before calculating activations:
        layer.set_weights([parameters[0]*scale_fac_prev_layer, parameters[1]])
        # t=4.9%
        get_activ = get_activ_fn_for_layer(model, idx)
        activations = get_activations_layer(get_activ, X_norm)
        if settings['normalization_schedule']:
            scale_fac = get_scale_fac(activations, idx)
        else:
            scale_fac = get_scale_fac(activations)  # t=3.7%
        parameters_norm = [parameters[0] * scale_fac_prev_layer / scale_fac,
                           parameters[1] / scale_fac]
        scale_fac_prev_layer = scale_fac
        # Update model with modified parameters
        layer.set_weights(parameters_norm)
        if settings['verbose'] < 3:
            continue
        weight_dict = {
            'weights': parameters[0].flatten(),
            'weights_norm': parameters_norm[0].flatten()}
        # t=2.8%
        plot_hist(weight_dict, 'Weight', layer.name, newpath)

        if True:  # Too costly
            continue
        # Compute activations with modified parameters
        # t=4.8%
        activations_norm = get_activations_layer(get_activ, X_norm)
        activation_dict = {'Activations': activations.flatten(),
                           'Activations_norm': activations_norm.flatten()}
        plot_hist(activation_dict, 'Activation', layer.name, newpath,
                  scale_fac)  # t=83.1%


def get_scale_fac(activations, idx=0):
    """Determine the maximum activation of a layer.

    Parameters
    ----------

    activations: array
        The activations of cells in a specific layer, flattened to 1-d.

    idx: int, optional
        The index of the layer. May be used to decrease the scale factor in
        higher layers, to maintain high spike rates.

    Returns
    -------

    scale_fac: float
        Maximum (or percentile) of activations in this layer.
        Parameters of the respective layer are scaled by this value.
    """

    scale_fac = np.percentile(activations, settings['percentile']-idx/10)
    if settings['verbose'] > 1:
        print("Scale factor: {:.2f}.".format(scale_fac))
    return scale_fac


def get_activations_layer(get_activ, X_train):
    """
    Get activations of a specific layer, iterating batch-wise over the complete
    data set.

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
    """Compute layer activations of an ANN.

    Parameters
    ----------

    ann: Keras model
        Needed to compute activations.

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
    for i, layer in enumerate(ann.layers):
        if 'Flatten' in layer.name:
            continue
        get_activ = get_activ_fn_for_layer(ann, i)
        activations_batch.append((get_activ(X_batch), layer.name))
    return activations_batch


def get_activ_fn_for_layer(model, i):
    f = theano.function(
        [model.layers[0].input, theano.In(K.learning_phase(), value=0)],
        model.layers[i].output, allow_input_downcast=True,
        on_unused_input='ignore')
    return lambda x: f(x).astype('float16', copy=False)


def wilson_score(p, n):
    """Confidence interval of a binomial distribution.

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
    """Get the layer number, name and shape from a string.

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
