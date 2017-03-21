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
import json
from importlib import import_module

import numpy as np
from future import standard_library
import keras
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

    # Parse input model to our common format, extracting all necessary
    # information about layers.
    model_lib = import_module('snntoolbox.model_libs.' +
                              settings['model_lib'] + '_input_lib')
    layers = model_lib.extract(input_model)

    # Create new Keras model
    img_input = keras.layers.Input(batch_shape=layers[0]['batch_input_shape'])
    parsed_layers = {'input_1': img_input}
    print("Building parsed model...")
    for layer in layers:
        # Replace 'parameters' key with Keras key 'weights'
        if 'parameters' in layer:
            layer['weights'] = layer.pop('parameters')
        # Remove keys that are not understood by Keras layer constructor
        layer_type = layer.pop('layer_type')
        filter_flip = layer.pop('filter_flip', None)
        if 'activation' in layer:
            a = layer['activation']
            if a == 'binary_sigmoid':
                layer['activation'] = binary_sigmoid
            elif a == 'binary_tanh':
                layer['activation'] = binary_tanh
            elif a == 'softmax' and settings['softmax_to_relu']:
                layer['activation'] = 'relu'
                print("Replaced softmax by relu activation function.")
        # Add layer
        parsed_layer = getattr(keras.layers, layer_type)
        if filter_flip:
            parsed_layer.filter_flip = filter_flip
        inbound = [parsed_layers[inb] for inb in layer.pop('inbound')]
        parsed_layers[layer['name']] = parsed_layer(**layer)(inbound)
    print("Compiling parsed model...")
    parsed_model = keras.models.Model(img_input,
                                      parsed_layers[layers[-1]['name']])
    # Optimizer and loss should not matter at this stage, but it would be
    # cleaner to set them to the actial values of the input model.
    parsed_model.compile('sgd', 'categorical_crossentropy', ['accuracy'])

    return parsed_model


def get_dataset(s):
    """Get data set.
    TODO: Docstring
    """

    evalset = normset = testset = None
    # Instead of loading a normalization set, try to get the scale-factors from
    # a previous run, stored on disk.
    newpath = os.path.join(settings['log_dir_of_current_run'], 'normalization')
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    filepath = os.path.join(newpath, str(settings['percentile']) + '.json')
    if os.path.isfile(filepath):
        print("Loading scale factors from disk instead of recalculating.")
        with open(filepath) as f:
            normset = {'scale_facs': json.load(f)}
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
        if s['normalize'] and normset is None:
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
        if 'batch_size' not in dataflow_kwargs:
            dataflow_kwargs['batch_size'] = s['batch_size']
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
        if s['normalize'] and normset is None:
            normset = {
                'dataflow': datagen.flow_from_directory(**dataflow_kwargs)}
            assert normset, "Normalization set empty."
        if s['simulate']:
            testset = {
                'dataflow': datagen.flow_from_directory(**dataflow_kwargs)}
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

    score = [0, 0]
    if x_test is not None:
        truth = np.argmax(y_test, axis=1)
        preds = model.predict(x_test, settings['batch_size'], verbose=0)
        score[0] = np.mean(np.argmax(preds, axis=1) == truth)
        score[1] = get_top5score(truth, preds) / len(truth)
    else:
        batches = int(settings['num_to_test'] / settings['batch_size'])
        for i in range(batches):
            # Get samples from Keras.ImageDataGenerator
            x_batch, y_batch = dataflow.next()
            if True:  # Only for imagenet!
                print("Preprocessing input for ImageNet")
                x_batch = np.add(np.multiply(x_batch, 2. / 255.), - 1.).astype(
                    'float32')
            truth = np.argmax(y_batch, axis=1)
            preds = model.predict_on_batch(x_batch)
            score[0] += np.mean(np.argmax(preds, axis=1) == truth)
            score[1] += get_top5score(truth, preds) / len(truth)
        score[0] /= batches
        score[1] /= batches
    print('\n' + "Top-1 accuracy: {:.2%}".format(score[0]))
    print("Top-5 accuracy: {:.2%}\n".format(score[1]))


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


def print_description(log=True):
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

    print("SNN-TOOLBOX SETTINGS", file=f)
    print("====================\n", file=f)
    print(settings, file=f)


def spikecounts_to_rates(spikecounts_n_b_l_t):
    """Convert spiketrains to spikerates.

        The output will have the same shape as the input except for the last
        dimension, which is removed by replacing a sequence of spiketimes by a
        single rate value.

        Parameters
        ----------

        spikecounts_n_b_l_t: list[tuple[np.array, str]]

        Returns
        -------

        : list[tuple[np.array, str]]
            spikerates_n_b_l
        """

    return [(np.mean(spikecounts_b_l_t, -1), name)
            for (spikecounts_b_l_t, name) in spikecounts_n_b_l_t]


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
                        np.count_nonzero(sp[0][ii][jj]) *
                        1000 / settings['duration'])
                    spikerates_batch[i][0][ii, jj] *= np.sign(
                        np.sum(sp[0][ii, jj]))  # For negative spikes
        elif len(shape) == 4:
            for ii in range(len(sp[0])):
                for jj in range(len(sp[0][ii])):
                    for kk in range(len(sp[0][ii, jj])):
                        for ll in range(len(sp[0][ii, jj, kk])):
                            spikerates_batch[i][0][ii, jj, kk, ll] = (
                                np.count_nonzero(sp[0][ii, jj, kk, ll]) *
                                1000 / settings['duration'])
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
    from collections import OrderedDict

    print("Normalizing parameters...")
    norm_dir = kwargs['path'] if 'path' in kwargs else \
        os.path.join(settings['log_dir_of_current_run'], 'normalization')
    activ_dir = os.path.join(norm_dir, 'activations')
    if not os.path.exists(activ_dir):
        os.makedirs(activ_dir)
    # Store original weights for later plotting
    if not os.path.isfile(os.path.join(activ_dir, 'weights.npz')):
        weights = {}
        for layer in model.layers:
            w = layer.get_weights()
            if len(w) > 0:
                weights[layer.name] = w[0]
        np.savez_compressed(os.path.join(activ_dir, 'weights.npz'), **weights)

    # Either load scale factors from disk, or get normalization data set to
    # calculate them.
    x_norm = None
    if 'scale_facs' in kwargs:
        scale_facs = kwargs['scale_facs']
    elif 'x_norm' in kwargs or 'dataflow' in kwargs:
        if 'x_norm' in kwargs:
            x_norm = kwargs['x_norm']
        elif 'dataflow' in kwargs:
            x_norm, y = kwargs['dataflow'].next()
            if True:  # Only for imagenet!
                print("Preprocessing input for ImageNet")
                x_norm = np.add(np.multiply(x_norm, 2. / 255.), - 1.)
        print("Using {} samples for normalization.".format(len(x_norm)))
        sizes = [
            len(x_norm) * np.array(layer.output_shape[1:]).prod() * 32 /
            (8*1e9) for layer in model.layers if len(layer.get_weights()) > 0]
        size_str = ['{:.2f}'.format(s) for s in sizes]
        print("INFO: Need {} GB for layer activations.\n".format(size_str) +
              "May have to reduce size of data set used for normalization.\n")
        scale_facs = OrderedDict({model.layers[0].name: 1})
    else:
        print("ERROR: No scale factors or normalization data set could not be "
              "loaded. Proceeding without normalization.")
        return

    # If scale factors have not been computed in a previous run, do so now.
    if len(scale_facs) == 1:
        from snntoolbox.io_utils.common import confirm_overwrite

        i = 0
        for layer in model.layers:
            # Skip if layer has no parameters
            if len(layer.get_weights()) == 0:
                continue

            print("Calculating activations of layer {} ...".format(
                layer.name, layer.output_shape))
            activations = get_activations_layer(model.input, layer.output,
                                                x_norm)
            if 'normalization_activations' in settings['plot_vars']:
                print("Writing activations to disk...")
                np.savez_compressed(os.path.join(activ_dir, layer.name),
                                    activations)
            nonzero_activations = activations[np.nonzero(activations)]
            del activations
            idx = i if settings['normalization_schedule'] else 0
            scale_facs[layer.name] = get_scale_fac(nonzero_activations, idx)
            # Since we have calculated output activations here, check at this
            # point if the output is mostly negative, in which case we should
            # stick to softmax. Otherwise ReLU is preferred.
            # Todo: Determine the input to the activation by replacing the
            # combined output layer by two distinct layers ``Dense`` and
            # ``Activation``!
            # if layer.activation == 'softmax' and settings['softmax_to_relu']:
            #     softmax_inputs = ...
            #     if np.median(softmax_inputs) < 0:
            #         print("WARNING: You allowed the toolbox to replace "
            #               "softmax by ReLU activations. However, more than "
            #               "half of the activations are negative, which could "
            #               "reduce accuracy. Consider setting "
            #               "settings['softmax_to_relu'] = False.")
            #         settings['softmax_to_relu'] = False
            i += 1
        # Write scale factors to disk
        filepath = os.path.join(norm_dir, str(settings['percentile']) + '.json')
        if confirm_overwrite(filepath):
            with open(filepath, 'w') as f:
                json.dump(scale_facs, f)

    # Apply scale factors to normalize the parameters.
    for layer in model.layers:
        # Skip if layer has no parameters
        if len(layer.get_weights()) == 0:
            continue

        # Scale parameters
        parameters = layer.get_weights()
        scale_fac = scale_facs[layer.name]
        inbound = get_inbound_layers_with_params(layer)
        if len(inbound) == 0:  # Input layer
            input_layer = layer.inbound_nodes[0].inbound_layers[0].name
            parameters_norm = [
                parameters[0] * scale_facs[input_layer] / scale_fac,
                parameters[1] / scale_fac]
        elif len(inbound) == 1:
            parameters_norm = [
                parameters[0] * scale_facs[inbound[0].name] / scale_fac,
                parameters[1] / scale_fac]
        else:
            parameters_norm = [parameters[0]]  # Consider only weights at first
            offset = 0  # Index offset at input filter dimension
            for inb in inbound:
                f_out = inb.W_shape[0]  # Num output features of inbound layer
                f_in = range(offset, offset + f_out)
                if parameters[0].ndim == 2:  # Fully-connected Layer
                    parameters_norm[0][f_in, :] *= \
                        scale_facs[inb.name] / scale_fac
                else:
                    parameters_norm[0][:, f_in, :, :] *= \
                        scale_facs[inb.name] / scale_fac
                offset += f_out
            parameters_norm.append(parameters[1] / scale_fac)  # Append bias

        # Update model with modified parameters
        layer.set_weights(parameters_norm)

    # Plot distributions of weights and activations before and after norm.
    if 'normalization_activations' in settings['plot_vars']:
        from snntoolbox.io_utils.plotting import plot_hist, plot_activ_hist
        from snntoolbox.io_utils.plotting import plot_max_activ_hist

        print("Plotting distributions of weights and activations before and "
              "after normalizing...")

        # Load original parsed model to get parameters before normalization
        weights = np.load(os.path.join(activ_dir, 'weights.npz'))
        for idx, layer in enumerate(model.layers):
            # Skip if layer has no parameters
            if len(layer.get_weights()) == 0:
                continue

            label = str(idx) + layer.__class__.__name__ if use_simple_label \
                else layer.name
            parameters = weights[layer.name]
            parameters_norm = layer.get_weights()
            weight_dict = {
                'weights': parameters[0].flatten(),
                'weights_norm': parameters_norm[0].flatten()}
            plot_hist(weight_dict, 'Weight', label, norm_dir)

            # Load activations of model before normalization
            activations = np.load(os.path.join(activ_dir,
                                               layer.name + '.npz'))['arr_0']
            # Compute activations with modified parameters
            nonzero_activations = activations[np.nonzero(activations)]
            activations_norm = get_activations_layer(model.input,
                                                     layer.output, x_norm)
            activation_dict = {
                'Activations': nonzero_activations, 'Activations_norm':
                    activations_norm[np.nonzero(activations_norm)]}
            scale_fac = scale_facs[layer.name]
            plot_hist(activation_dict, 'Activation', label, norm_dir,
                      scale_fac)
            ax = tuple(np.arange(len(layer.output_shape))[1:])
            plot_activ_hist({'Activations': nonzero_activations},
                            'Activation', label, norm_dir, scale_fac)
            plot_max_activ_hist(
                {'Activations_max': np.max(activations, axis=ax)},
                'Maximum Activation', label, norm_dir, scale_fac)
    print()


def get_inbound_layers_with_params(layer):
    """Iterate until inbound layers are found that have parameters.

    Parameters
    ----------

    layer: keras.layers.Layer
        Layer

    Returns
    -------

    : list
        List of inbound layers.
    """

    inbound = layer
    while True:
        inbound = get_inbound_layers(inbound)
        if len(inbound) == 1:
            inbound = inbound[0]
            if len(inbound.get_weights()) > 0:
                return [inbound]
        else:
            result = []
            for inb in inbound:
                if len(inb.get_weights()) > 0:
                    result.append(inb)
                else:
                    result += get_inbound_layers_with_params(inb)
            return result


def get_inbound_layers_without_params(layer):
    """Return inbound layers.

    Parameters
    ----------

    layer: Keras.layers
        A Keras layer.

    Returns
    -------

    : list[Keras.layers]
        List of inbound layers.
    """

    return [layer for layer in layer.inbound_nodes[0].inbound_layers
            if len(layer.get_weights()) == 0]


def get_inbound_layers(layer):
    """Return inbound layers.

    Parameters
    ----------

    layer: Keras.layers
        A Keras layer.

    Returns
    -------

    : list[Keras.layers]
        List of inbound layers.
    """

    return layer.inbound_nodes[0].inbound_layers


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

    scale_fac = np.percentile(activations, settings['percentile'] - idx * 0.02)
    print("Scale factor: {:.2f}.".format(scale_fac))

    return scale_fac


def get_activations_layer(layer_in, layer_out, x):
    """
    Get activations of a specific layer, iterating batch-wise over the complete
    data set.

    Parameters
    ----------

    layer_in: keras.layers.Layer
        The input to the network.

    layer_out: keras.layers.Layer
        The layer for which we want to get the activations.

    x: np.array
        The samples to compute activations for. With data of the form
        (channels, num_rows, num_cols), x_train has dimension
        (batch_size, channels*num_rows*num_cols) for a multi-layer perceptron,
        and (batch_size, channels, num_rows, num_cols) for a convolutional net.

    Returns
    -------

    activations: np.array
        The activations of cells in a specific layer. Has the same shape as
        ``layer_out``.
    """

    return keras.models.Model(layer_in, layer_out).predict(x)


def get_activations_batch(ann, x_batch):
    """Compute layer activations of an ANN.

    Parameters
    ----------

    ann: keras.models.Model
        Needed to compute activations.

    x_batch: np.array
        The input samples to use for determining the layer activations. With
        data of the form (channels, num_rows, num_cols), X has dimension
        (batch_size, channels*num_rows*num_cols) for a multi-layer perceptron,
        and (batch_size, channels, num_rows, num_cols) for a convolutional net.

    Returns
    -------

    activations_batch: list[tuple[np.array, str]]
        Each tuple ``(activations, label)`` represents a layer in the ANN for
        which an activation can be calculated (e.g. ``Dense``,
        ``Convolution2D``).
        ``activations`` containing the activations of a layer. It has the same
        shape as the original layer, e.g.
        (batch_size, n_features, n_rows, n_cols) for a convolution layer.
        ``label`` is a string specifying the layer type, e.g. ``'Dense'``.
    """

    activations_batch = []
    for layer in ann.layers:
        if layer.__class__.__name__ in ['Input', 'InputLayer', 'Flatten',
                                        'Merge']:
            continue
        activations = keras.models.Model(ann.input,
                                         layer.output).predict_on_batch(x_batch)
        activations_batch.append((activations, layer.name))
    return activations_batch


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


def get_top5score(truth, output):
    """Compute the top-5-score (not averaged).

    Parameters
    ----------

    truth: np.array
        Target classes.
    output: np.array
        Output of final classification layer. Shape: (batch_size, num_classes).

    Returns
    -------

    score: float
        The top-5-score (not averaged over samples).
    """

    score = 0
    for t, o in zip(truth, output):
        o = np.array(o, 'float32')
        top5pred = []
        for i in range(5):
            top = np.argmax(o)
            top5pred.append(top)
            o[top] = -np.infty
        if t in top5pred:
            score += 1
    return score

# python 2 can not handle the 'flush' keyword argument of python 3 print().
# Provide 'echo' function as an alias for
# "print with flush and without newline".
try:
    from functools import partial
    echo = partial(print, end='', flush=True)
    echo(u'')
except TypeError:
    # TypeError: 'flush' is an invalid keyword argument for this function
    import sys

    def echo(text):
        """python 2 version of print(end='', flush=True)."""
        sys.stdout.write(u'{0}'.format(text))
        sys.stdout.flush()


def to_list(x):
    """Normalize a list/tensor to a list.

    If a tensor is passed, returns a list of size 1 containing the tensor.
    """

    return x if type(x) is list else [x]
