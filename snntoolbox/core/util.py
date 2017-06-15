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
import numpy as np
import keras
from future import standard_library

standard_library.install_aliases()


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

    t = spikecounts_n_b_l_t[0][0].shape[-1] + 1

    return [(np.true_divide(spikecounts_b_l_t[Ellipsis, -1], t), name)
            for (spikecounts_b_l_t, name) in spikecounts_n_b_l_t]


def spiketrains_to_rates(spiketrains_n_b_l_t, duration):
    """Convert spiketrains to spikerates.

    The output will have the same shape as the input except for the last
    dimension, which is removed by replacing a sequence of spiketimes by a
    single rate value.

    Parameters
    ----------

    spiketrains_n_b_l_t: list[tuple[np.array, str]]

    duration: int
        Duration of simulation.

    Returns
    -------

    spikerates_n_b_l: list[tuple[np.array, str]]
    """

    spikerates_n_b_l = []
    for (i, sp) in enumerate(spiketrains_n_b_l_t):
        shape = sp[0].shape[:-1]  # output_shape of layer
        # Allocate list containing an empty array of shape
        # 'output_shape' for each layer of the network, which will
        # hold the spikerates of a mini-batch.
        spikerates_n_b_l.append((np.empty(shape), sp[1]))
        # Count number of spikes fired in the layer and divide by the
        # simulation time to get the mean firing rate of each neuron.
        if len(shape) == 2:
            for ii in range(len(sp[0])):
                for jj in range(len(sp[0][ii])):
                    spikerates_n_b_l[i][0][ii, jj] = (
                        np.count_nonzero(sp[0][ii, jj]) / duration)
                    spikerates_n_b_l[i][0][ii, jj] *= np.sign(
                        np.sum(sp[0][ii, jj]))  # For negative spikes
        elif len(shape) == 4:
            for ii in range(len(sp[0])):
                for jj in range(len(sp[0][ii])):
                    for kk in range(len(sp[0][ii, jj])):
                        for ll in range(len(sp[0][ii, jj, kk])):
                            spikerates_n_b_l[i][0][ii, jj, kk, ll] = (
                                np.count_nonzero(sp[0][ii, jj, kk, ll]) /
                                duration)
                            spikerates_n_b_l[i][0][ii, jj, kk, ll] *= np.sign(
                                np.sum(sp[0][ii, jj, kk, ll]))

    return spikerates_n_b_l


def get_sample_activity_from_batch(activity_batch, idx=0):
    """Return layer activity for sample ``idx`` of an ``activity_batch``.
    """

    return [(layer_act[0][idx], layer_act[1]) for layer_act in activity_batch]


def try_reload_activations(layer, model, x_norm, batch_size, activ_dir):
    try:
        activations = np.load(os.path.join(activ_dir,
                                           layer.name + '.npz'))['arr_0']
    except FileNotFoundError:
        if x_norm is None:
            return

        print("Calculating activations of layer {} ...".format(layer.name))
        activations = get_activations_layer(model.input, layer.output, x_norm,
                                            batch_size)
        print("Writing activations to disk...")
        np.savez_compressed(os.path.join(activ_dir, layer.name), activations)

    return activations


def normalize_parameters(model, config, **kwargs):
    """Normalize the parameters of a network.

    The parameters of each layer are normalized with respect to the maximum
    activation, or the ``n``-th percentile of activations.

    Generates plots of the activity- and weight-distribution before and after
    normalization. Note that plotting the activity-distribution can be very
    time- and memory-consuming for larger networks.
    """

    import json
    from collections import OrderedDict

    print("Normalizing parameters...")
    norm_dir = kwargs[str('path')] if 'path' in kwargs else \
        os.path.join(config['paths']['log_dir_of_current_run'], 'normalization')
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
        scale_facs = kwargs[str('scale_facs')]
    elif 'x_norm' in kwargs or 'dataflow' in kwargs:
        if 'x_norm' in kwargs:
            x_norm = kwargs[str('x_norm')]
        elif 'dataflow' in kwargs:
            x_norm, y = kwargs[str('dataflow')].next()
        print("Using {} samples for normalization.".format(len(x_norm)))
        sizes = [
            len(x_norm) * np.array(layer.output_shape[1:]).prod() * 32 /
            (8*1e9) for layer in model.layers if len(layer.weights) > 0]
        size_str = ['{:.2f}'.format(s) for s in sizes]
        print("INFO: Need {} GB for layer activations.\n".format(size_str) +
              "May have to reduce size of data set used for normalization.")
        scale_facs = OrderedDict({model.layers[0].name: 1})
    else:
        print("ERROR: No scale factors or normalization data set could not be "
              "loaded. Proceeding without normalization.")
        return

    batch_size = config.getint('simulation', 'batch_size')

    # If scale factors have not been computed in a previous run, do so now.
    if len(scale_facs) == 1:
        from snntoolbox.io_utils.common import confirm_overwrite

        i = 0
        for layer in model.layers:
            # Skip if layer has no parameters
            if len(layer.weights) == 0:
                continue

            activations = try_reload_activations(layer, model, x_norm,
                                                 batch_size, activ_dir)
            nonzero_activations = activations[np.nonzero(activations)]
            del activations
            perc = get_percentile(config, i)
            scale_facs[layer.name] = get_scale_fac(nonzero_activations, perc)
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
        filepath = os.path.join(norm_dir, config['normalization']['percentile']
                                + '.json')
        if config['output']['overwrite'] or confirm_overwrite(filepath):
            with open(filepath, str('w')) as f:
                json.dump(scale_facs, f)

    # Apply scale factors to normalize the parameters.
    for layer in model.layers:
        # Skip if layer has no parameters
        if len(layer.weights) == 0:
            continue

        # Scale parameters
        parameters = layer.get_weights()
        if layer.activation.__name__ == 'softmax':
            # When using a certain percentile or even the max, the scaling
            # factor can be extremely low in case of many output classes
            # (e.g. 0.01 for ImageNet). This amplifies weights and biases
            # greatly. But large biases cause large offsets in the beginning
            # of the simulation (spike input absent).
            scale_fac = 1.0
            print("Using scale factor {:.2f} for softmax layer.".format(
                scale_fac))
        else:
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
                f_out = inb.filters  # Num output features of inbound layer
                f_in = range(offset, offset + f_out)
                if parameters[0].ndim == 2:  # Fully-connected Layer
                    parameters_norm[0][f_in, :] *= \
                        scale_facs[inb.name] / scale_fac
                else:
                    parameters_norm[0][:, :, f_in, :] *= \
                        scale_facs[inb.name] / scale_fac
                offset += f_out
            parameters_norm.append(parameters[1] / scale_fac)  # Append bias

        # Update model with modified parameters
        layer.set_weights(parameters_norm)

    # Plot distributions of weights and activations before and after norm.
    if 'normalization_activations' in eval(config['output']['plot_vars']):
        from snntoolbox.io_utils.plotting import plot_hist, plot_activ_hist
        from snntoolbox.io_utils.plotting import plot_max_activ_hist

        print("Plotting distributions of weights and activations before and "
              "after normalizing...")

        # Load original parsed model to get parameters before normalization
        weights = np.load(os.path.join(activ_dir, 'weights.npz'))
        for idx, layer in enumerate(model.layers):
            # Skip if layer has no parameters
            if len(layer.weights) == 0:
                continue

            label = str(idx) + layer.__class__.__name__ if \
                config.getboolean('output', 'use_simple_labels') else layer.name
            parameters = weights[layer.name]
            parameters_norm = layer.get_weights()
            weight_dict = {
                'weights': parameters[0].flatten(),
                'weights_norm': parameters_norm[0].flatten()}
            plot_hist(weight_dict, 'Weight', label, norm_dir)

            # Load activations of model before normalization
            activations = try_reload_activations(layer, model, x_norm,
                                                 batch_size, activ_dir)

            if activations is None:
                continue

            # Compute activations with modified parameters
            nonzero_activations = activations[np.nonzero(activations)]
            activations_norm = get_activations_layer(model.input, layer.output,
                                                     x_norm, batch_size)
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
    print('')


def has_weights(layer):
    """Return ``True`` if layer has weights.

    Parameters
    ----------

    layer : keras.layers.Layer
        Keras layer

    Returns
    -------

    : bool
        ``True`` if layer has weights.
    """

    return len(layer.weights)


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
            if has_weights(inbound) > 0:
                return [inbound]
        else:
            result = []
            for inb in inbound:
                if has_weights(inb):
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
            if len(layer.weights) == 0]


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


def get_outbound_layers(layer):
    """Return outbound layers.

    Parameters
    ----------

    layer: Keras.layers
        A Keras layer.

    Returns
    -------

    : list[Keras.layers]
        List of outbound layers.
    """

    return [on.outbound_layer for on in layer.outbound_nodes]


def get_outbound_activation(layer):
    """
    Iterate over 2 outbound layers to find an activation layer. If there is no
    activation layer, take the activation of the current layer.

    Parameters
    ----------

    layer: Union[keras.layers.Conv2D, keras.layers.Dense]
        Layer

    Returns
    -------

    activation: str
        Name of outbound activation type.
    """

    activation = layer.activation.__name__
    outbound = layer
    for _ in range(2):
        outbound = get_outbound_layers(outbound)
        if len(outbound) == 1 and get_type(outbound[0]) == 'Activation':
            activation = outbound[0].activation.__name__
    return activation


def get_spiking_outbound_layers(layer):
    """Iterate until spiking outbound layers are found.

    Parameters
    ----------

    layer: keras.layers.Layer
        Layer

    Returns
    -------

    : list
        List of outbound layers.
    """

    outbound = layer
    while True:
        outbound = get_outbound_layers(outbound)
        if len(outbound) == 1:
            outbound = outbound[0]
            if is_spiking(outbound):
                return [outbound]
        else:
            result = []
            for outb in outbound:
                if is_spiking(outb):
                    result.append(outb)
                else:
                    result += get_spiking_outbound_layers(outb)
            return result


def get_scale_fac(activations, percentile):
    """
    Determine the activation value at ``percentile`` of the layer distribution.

    Parameters
    ----------

    activations: np.array
        The activations of cells in a specific layer, flattened to 1-d.

    percentile: int
        Percentile at which to determine activation.

    Returns
    -------

    scale_fac: float
        Maximum (or percentile) of activations in this layer.
        Parameters of the respective layer are scaled by this value.
    """

    scale_fac = np.percentile(activations, percentile)
    print("Scale factor: {:.2f}.".format(scale_fac))

    return scale_fac


def get_percentile(config, layer_idx):
    """Get percentile at which to draw the maximum activation of a layer.

    Parameters
    ----------

    config: configparser.ConfigParser
        Settings.

    layer_idx: int
        Layer index.

    Returns
    -------

    : int
        Percentile.

    """

    perc = config.getfloat('normalization', 'percentile')

    if config.getboolean('normalization', 'normalization_schedule'):
        perc = apply_normalization_schedule(perc, layer_idx)

    return perc


def apply_normalization_schedule(perc, layer_idx):
    """Transform percentile according to some rule, depending on layer index.

    Parameters
    ----------

    perc: float
        Original percentile.

    layer_idx: int
        Layer index, used to decrease the scale factor in higher layers, to
        maintain high spike rates.

    Returns
    -------

    : int
        Modified percentile.

    """

    return int(perc - layer_idx * 0.02)


def get_activations_layer(layer_in, layer_out, x, batch_size=None):
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

    batch_size: Optional[int]
        Batch size

    Returns
    -------

    activations: np.array
        The activations of cells in a specific layer. Has the same shape as
        ``layer_out``.
    """

    kwargs = {} if batch_size is None else {'batch_size': batch_size}

    return keras.models.Model(layer_in, layer_out).predict(x, **kwargs)


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
        ``Conv2D``).
        ``activations`` containing the activations of a layer. It has the same
        shape as the original layer, e.g.
        (batch_size, n_features, n_rows, n_cols) for a convolution layer.
        ``label`` is a string specifying the layer type, e.g. ``'Dense'``.
    """

    activations_batch = []
    for layer in ann.layers:
        if layer.__class__.__name__ in ['Input', 'InputLayer', 'Flatten',
                                        'Concatenate']:
            continue
        activations = keras.models.Model(ann.input,
                                         layer.output).predict_on_batch(x_batch)
        activations_batch.append((activations, layer.name))
    return activations_batch


def get_log_keys(config):
    return set(eval(config['output']['log_vars']))


def get_plot_keys(config):
    return set(eval(config['output']['plot_vars']))


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

    label: str
        Specifies both the layer type, index and shape, e.g.
        ``'03Conv2D_3x32x32'``.

    Returns
    -------

    : tuple[int, str, tuple]
        - layer_num: The index of the layer in the network.
        - name: The type of the layer.
        - shape: The shape of the layer
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


def in_top_k(predictions, targets, k):
    """Returns whether the `targets` are in the top `k` `predictions`

    # Arguments
        predictions: A tensor of shape batch_size x classess and type float32.
        targets: A tensor of shape batch_size and type int32 or int64.
        k: An int, number of top elements to consider.

    # Returns
        A tensor of shape batch_size and type int. output_i is 1 if
        targets_i is within top-k values of predictions_i
    """

    predictions_top_k = np.argsort(predictions)[:, -k:]
    return np.array([np.equal(p, t).any() for p, t in zip(predictions_top_k,
                                                          targets)])


def top_k_categorical_accuracy(y_true, y_pred, k=5):
    """

    Parameters
    ----------
    y_true : 
    y_pred : 
    k : 

    Returns
    -------

    """

    return np.mean(in_top_k(y_pred, np.argmax(y_true, axis=-1), k))


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


def get_layer_ops(spiketrains_b_l, fanout, num_neurons_with_bias=0):
    """
    Return total number of operations in the layer for a batch of samples.  

    Parameters
    ----------

    spiketrains_b_l: ndarray
        Batch of spiketrains of a layer. Shape: (batch_size, layer_shape)
    fanout: int
        Number of outgoing connections per neuron.
    num_neurons_with_bias: int
        Number of neurons with bias.
    """

    return np.array([np.count_nonzero(s) for s in spiketrains_b_l]) * fanout + \
        num_neurons_with_bias


def get_fanin(layer):
    """
    Return fan-in of a neuron in ``layer``.

    Parameters
    ----------

    layer: Subclass[keras.layers.Layer] 
         Layer.

    Returns
    -------

    fanin: int
        Fan-in.

    """

    if 'Conv' in layer.name:
        fanin = np.prod(layer.kernel_size) * layer.input_shape[1]
    elif 'Dense' in layer.name:
        fanin = layer.input_shape[1]
    elif 'Pool' in layer.name:
        fanin = 0
    else:
        fanin = 0

    return fanin


def get_fanout(layer):
    """
    Return fan-out of a neuron in ``layer``.

    Parameters
    ----------

    layer: Subclass[keras.layers.Layer] 
         Layer.

    Returns
    -------

    fanout: int
        Fan-out.

    """

    fanout = 0
    next_layers = get_spiking_outbound_layers(layer)
    for next_layer in next_layers:
        if 'Conv' in layer.name and 'Pool' in next_layer.name:
            fanout += 1
        elif 'Dense' in layer.name:
            fanout += next_layer.units
        elif 'Pool' in layer.name and 'Conv' in next_layer.name:
            fanout += np.prod(next_layer.kernel_size) * next_layer.filters
        elif 'Pool' in layer.name and 'Dense' in next_layer.name:
            fanout += next_layer.units
        else:
            fanout += 0

    return fanout


def is_spiking(layer):
    """Test if layer is going to be converted to a layer that spikes.

    Parameters
    ----------

    layer: Keras.layers.Layer
        Layer of parsed model.

    Returns
    -------

    : boolean
        ``True`` if converted layer will have spiking neurons.

    """

    from snntoolbox.config import spiking_layers

    return get_type(layer) in spiking_layers


def get_ann_ops(num_neurons, num_neurons_with_bias, fanin):
    """
    Compute number of operations performed by an ANN in one forward pass.

    Parameters
    ----------

    num_neurons: list[int]
        Number of neurons per layer, starting with input layer.
    num_neurons_with_bias: list[int]
        Number of neurons with bias.
    fanin: list[int]
        List of fan-in of neurons in Conv, Dense and Pool layers. Input and Pool
        layers have fan-in 0 so they are not counted.


    Returns
    -------

    : int
        Number of operations.

    """

    return 2 * np.dot(num_neurons, fanin) + np.sum(num_neurons_with_bias)


def get_type(layer):
    """Get type of Keras layer.

    Parameters
    ----------

    layer: Keras.layers.Layer
        Keras layer.

    Returns
    -------

    : str
        Layer type.

    """

    return layer.__class__.__name__
