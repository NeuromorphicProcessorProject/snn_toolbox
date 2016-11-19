# -*- coding: utf-8 -*-
"""
Helper functions to handle parameters and variables of interest during
conversion and simulation of an SNN.

Created on Wed Mar  9 16:18:33 2016

@author: rbodo
"""

# For compatibility with python2
from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import os
from importlib import import_module

import numpy as np
import theano
from future import standard_library
from keras import backend as k
from snntoolbox.config import settings

standard_library.install_aliases()

use_simple_label = True


def get_root_dir():
    """Get toolbox root directory.

    Returns
    -------

    : str
        Toolbox root directory.
    """

    return os.getcwd()


def binary_tanh(x):
    """Round a float to -1 or 1.

    Parameters
    ----------

    x: float

    Returns
    -------

    : int
        Integer in {-1, 1}
    """

    return k.sign(x)


def binary_sigmoid(x):
    """Round a float to 0 or 1.

    Parameters
    ----------

    x: float

    Returns
    -------

    : int
        Integer in {0, 1}
    """

    x = k.clip((x + 1.) / 2., 0, 1)

    return k.round(x)


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

    input_model: Any
        A pretrained neural network model in the respective ``model_lib``.

    Returns
    -------

    parsed_model: keras.models.Sequential
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
        if 'activation' in layer:
            if layer['activation'] == 'binary_sigmoid':
                layer['activation'] = binary_sigmoid
            elif layer['activation'] == 'binary_tanh':
                layer['activation'] = binary_tanh
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


def get_dataset(s):
    """Get data set.
    TODO: Docstring
    """

    evalset = normset = testset = None
    if s['dataset_format'] == 'npz':
        print("Loading data set from '.npz' files in {}.\n".format(
            s['dataset_path']))
        from snntoolbox.io_utils.common import load_dataset
        if s['evaluateANN'] or s['simulate']:
            evalset = {
                'x_test': load_dataset(s['dataset_path'], 'x_test.npz'),
                'y_test': load_dataset(s['dataset_path'], 'y_test.npz')}
#            # Binarize the input. Hack: Should be independent of maxpool type
#            if s['maxpool_type'] == 'binary_tanh':
#                evalset['x_test'] = np.sign(evalset['x_test'])
#            elif s['maxpool_type'] == 'binary_sigmoid':
#                np.clip((evalset['x_test']+1.)/2., 0, 1, evalset['x_test'])
#                np.round(evalset['x_test'], out=evalset['x_test'])
            assert evalset, "Evaluation set empty."
        if s['normalize']:
            normset = {
                'x_norm': load_dataset(s['dataset_path'], 'x_norm.npz')}
            assert normset, "Normalization set empty."
        if s['simulate']:
            testset = evalset
            assert testset, "Test set empty."
    elif s['dataset_format'] == 'jpg':
        import ast
        from keras.preprocessing.image import ImageDataGenerator
        print("Loading data set from ImageDataGenerator, using images in "
              "{}.\n".format(s['dataset_path']))
        datagen_kwargs = ast.literal_eval(s['datagen_kwargs'])
        dataflow_kwargs = ast.literal_eval(s['dataflow_kwargs'])
        dataflow_kwargs['directory'] = s['dataset_path']
        dataflow_kwargs['batch_size'] = s['num_to_test']
        datagen = ImageDataGenerator(**datagen_kwargs)
        # Compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied)
        rs = datagen_kwargs['rescale'] if 'rescale' in datagen_kwargs else None
        x_orig = ImageDataGenerator(rescale=rs).flow_from_directory(
            **dataflow_kwargs).next()[0]
        datagen.fit(x_orig)
        if s['evaluateANN']:
            evalset = {
                'dataflow': datagen.flow_from_directory(**dataflow_kwargs)}
            assert evalset, "Evaluation set empty."
        if s['normalize']:
            batchflow_kwargs = dataflow_kwargs.copy()
            batchflow_kwargs['batch_size'] = s['batch_size']
            normset = {
                'dataflow': datagen.flow_from_directory(**batchflow_kwargs)}
            assert normset, "Normalization set empty."
        if s['simulate']:
            batchflow_kwargs = dataflow_kwargs.copy()
            batchflow_kwargs['batch_size'] = s['batch_size']
            testset = {
                'dataflow': datagen.flow_from_directory(**batchflow_kwargs)}
            assert testset, "Test set empty."
    return evalset, normset, testset


def evaluate_keras(model, x_test=None, y_test=None, dataflow=None):
    """Evaluate parsed Keras model.

    Can use either numpy arrays ``x_test, y_test`` containing the test samples,
    or generate them with a dataflow
    (``Keras.ImageDataGenerator.flow_from_directory`` object).
    """

    assert (
        x_test is not None and y_test is not None or dataflow is not None), \
        "No testsamples provided."

    if dataflow:
        batch_size = dataflow.batch_size
        dataflow.batch_size = settings['num_to_test']
        score = model.evaluate_generator(dataflow, settings['num_to_test'])
        dataflow.batch_size = batch_size
    else:
        score = model.evaluate(x_test, y_test)
    print('\n' + "Test loss: {:.2f}".format(score[0]))
    print("Test accuracy: {:.2%}\n".format(score[1]))
    return score


def get_range(start=0.0, stop=1.0, num=5, method='linear'):
    """Return a range of parameter values.

    Convenience function. For more flexibility, use ``numpy.linspace``,
    ``numpy.logspace``, ``numpy.random.random_sample`` directly.

    Parameters
    ----------

    start: float
        The starting value of the sequence
    stop: float
        End value of the sequence.
    num: int
        Number of samples to generate. Must be non-negative.
    method: str
        The sequence will be computed on either a linear, logarithmic or random
        grid.

    Returns
    -------

    samples: np.array
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
        return np.random.random_sample(num) * (stop - start) + start


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

    Parameters
    ----------

    spiketrains_batch: list[tuple[np.array, str]]

    Returns
    -------

    spikerates_batch: list[tuple[np.array, str]]
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
                        np.count_nonzero(sp[0][ii][jj]) * 1000 *
                        settings['dt'] / settings['duration'])
                    spikerates_batch[i][0][ii, jj] *= np.sign(
                        np.sum(sp[0][ii, jj]))  # For negative spikes
        elif len(shape) == 4:
            for ii in range(len(sp[0])):
                for jj in range(len(sp[0][ii])):
                    for kk in range(len(sp[0][ii, jj])):
                        for ll in range(len(sp[0][ii, jj, kk])):
                            spikerates_batch[i][0][ii, jj, kk, ll] = (
                                np.count_nonzero(sp[0][ii, jj, kk, ll]) * 1000 *
                                settings['dt'] / settings['duration'])
                            spikerates_batch[i][0][ii, jj, kk, ll] *= np.sign(
                                np.sum(sp[0][ii, jj, kk, ll]))

    return spikerates_batch


def get_sample_activity_from_batch(activity_batch, idx=0):
    """Return layer activity for sample ``idx`` of an ``activity_batch``.
    """

    return [(layer_act[0][idx], layer_act[1]) for layer_act in activity_batch]


def normalize_parameters(model, **kwargs):
    """Normalize the parameters of a network.

    The parameters of each layer are normalized with respect to the maximum
    activation, or the ``n``-th percentile of activations.

    Generates plots of the activity- and weight-distribution before and after
    normalization. Note that plotting the activity-distribution can be very
    time- and memory-consuming for larger networks.
    """

    import json
    from snntoolbox.io_utils.plotting import plot_hist
    from snntoolbox.io_utils.common import confirm_overwrite

    assert 'x_norm' in kwargs or 'dataflow' in kwargs, \
        "Normalization data set could not be loaded."
    x_norm = None
    if 'x_norm' in kwargs:
        x_norm = kwargs['x_norm']
    elif 'dataflow' in kwargs:
        x_norm, y = kwargs['dataflow'].next()

    print("Using {} samples for normalization.".format(len(x_norm)))

    #        import numpy as np
    #        sizes = [len(x_norm) * np.array(layer['output_shape'][1:]).prod() *
    #                 32 / (8 * 1e9) for idx, layer in enumerate(self.layers)
    #                 if idx != 0 and 'parameters' in self.layers[idx-1]]
    #        size_str = ['{:.2f}'.format(s) for s in sizes]
    #        print("INFO: Need {} GB for layer activations.\n".format(
    # size_str) +
    #              "May have to reduce size of data set used for
    # normalization.\n")

    print("Normalizing parameters:\n")
    newpath = kwargs['path'] if 'path' in kwargs else \
        os.path.join(settings['log_dir_of_current_run'], 'normalization')
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    filepath = os.path.join(newpath, str(settings['percentile']) + '.json')
    if os.path.isfile(filepath) and os.path.basename(filepath) == str(
            settings['percentile']) + '.json':
        print("Loading scale factors from disk instead of recalculating.")
        facs_from_disk = True
        with open(filepath) as f:
            scale_facs = json.load(f)
    else:
        facs_from_disk = False
        scale_facs = []

    # Loop through all layers, looking for layers with parameters
    i = 0
    get_activ = None
    scale_fac_prev_layer = 1
    for idx, layer in enumerate(model.layers):
        # Skip layer if not preceeded by a layer with parameters
        if len(layer.get_weights()) == 0:
            continue
        parameters = layer.get_weights()

        if facs_from_disk:
            scale_fac_prev_layer = scale_facs[i - 1] if i > 0 else 1
            scale_fac = scale_facs[i]
            i += 1
        else:
            if settings['verbose'] > 1:
                print("Calculating activation of layer {} ...".format(
                    layer.name, layer.output_shape))
            # Undo previous scaling before calculating activations:
            layer.set_weights([parameters[0] * scale_fac_prev_layer,
                               parameters[1]])
            # t=4.9%
            get_activ = get_activ_fn_for_layer(model, idx)
            activations = get_activations_layer(get_activ, x_norm)
            if settings['normalization_schedule']:
                scale_fac = get_scale_fac(activations, idx)
            else:
                scale_fac = get_scale_fac(activations)  # t=3.7%
            scale_facs.append(scale_fac)

        # Scale parameters
        parameters_norm = [parameters[0] * scale_fac_prev_layer / scale_fac,
                           parameters[1] / scale_fac]
        scale_fac_prev_layer = scale_fac

        # Update model with modified parameters
        layer.set_weights(parameters_norm)

        # Plot distributions of weights and activations before and after norm.
        if facs_from_disk:
            continue  # Assume plots are already there.
        if settings['verbose'] < 3:
            continue
        label = str(idx) + layer.__class__.__name__ if use_simple_label \
            else layer.name
        weight_dict = {
            'weights': parameters[0].flatten(),
            'weights_norm': parameters_norm[0].flatten()}
        # t=2.8%
        plot_hist(weight_dict, 'Weight', label, newpath)

        # Compute activations with modified parameters
        # t=4.8%
        activations_norm = get_activations_layer(get_activ, x_norm)
        activation_dict = {'Activations': activations[np.nonzero(activations)],
                           'Activations_norm':
                               activations_norm[np.nonzero(activations)]}
        plot_hist(activation_dict, 'Activation', label, newpath, scale_fac)
    # plot_hist({'Activations': activations[np.nonzero(activations)]},
    #                  'Activation', label, newpath, scale_fac)
    #        plot_hist({'Activations_max': np.max(activations, axis=tuple(
    #            np.arange(activations.ndim)[1:]))}, 'Activation', label,
    # newpath,
    #            scale_fac)
    # t=83.1%
    # Write scale factors to disk
    if not facs_from_disk and confirm_overwrite(filepath):
        with open(filepath, 'w') as f:
            json.dump(scale_facs, f)


def get_scale_fac(activations, idx=0):
    """Determine the maximum activation of a layer.

    Parameters
    ----------

    activations: np.array
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

    # Remove zeros, because they bias the distribution too much
    a = activations[np.nonzero(activations)]

    scale_fac = np.percentile(a, settings['percentile'] - idx ** 2 / 200,
                              overwrite_input=True)
    if settings['verbose'] > 1:
        print("Scale factor: {:.2f}.".format(scale_fac))

    return scale_fac


def get_activations_layer(get_activ, x_train):
    """
    Get activations of a specific layer, iterating batch-wise over the complete
    data set.

    Parameters
    ----------

    get_activ: Theano function
        A Theano function computing the activations of a layer.

    x_train: float32 array
        The samples to compute activations for. With data of the form
        (channels, num_rows, num_cols), x_train has dimension
        (batch_size, channels*num_rows*num_cols) for a multi-layer perceptron,
        and (batch_size, channels, num_rows, num_cols) for a convolutional net.

    Returns
    -------

    activations: np.array
        The activations of cells in a specific layer. Has the same shape as the
        layer.
    """

    shape = list(get_activ(x_train[:settings['batch_size']]).shape)
    shape[0] = x_train.shape[0]
    activations = np.empty(shape)
    num_batches = int(np.ceil(x_train.shape[0] / settings['batch_size']))
    for batch_idx in range(num_batches):
        # Determine batch indices.
        max_idx = min((batch_idx + 1) * settings['batch_size'],
                      x_train.shape[0])
        batch_idxs = range(batch_idx * settings['batch_size'], max_idx)
        batch = x_train[batch_idxs, :]
        if len(batch_idxs) < settings['batch_size']:
            batch.resize(x_train[:settings['batch_size']].shape)
            activations[batch_idxs] = get_activ(batch)[:len(batch_idxs)]
        else:
            activations[batch_idxs] = get_activ(batch)
    return activations


def get_activations_batch(ann, x_batch):
    """Compute layer activations of an ANN.

    Parameters
    ----------

    ann: Keras model
        Needed to compute activations.

    x_batch: float32 array
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
        activations_batch.append((get_activ(x_batch), layer.name))
    return activations_batch


def get_activ_fn_for_layer(model, i):
    """Get a function that computes the activations of a layer.

    :param model: The network.
    :param i: The layer index.
    :return: A theano function that computes the activations of layer ``i``.
    """

    f = theano.function(
        [model.layers[0].input, theano.In(k.learning_phase(), value=0)],
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
    return (z * np.sqrt((p * (1 - p) + z * z / (4 * n)) / n)) / (1 + z * z / n)


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
    return layer_num, name, shape
