# -*- coding: utf-8 -*-
"""

This module performs modifications on the network parameters during conversion
from analog to spiking.

.. autosummary::
    :nosignatures:

    normalize_parameters

@author: rbodo
"""

from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import os

import keras
import numpy as np
from future import standard_library

standard_library.install_aliases()


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
    from snntoolbox.parsing.utils import get_inbound_layers_with_params
    from snntoolbox.parsing.utils import get_inbound_layers_with_params, get_inbound_layers, get_fanin, get_outbound_layers
    from snntoolbox.parsing.utils import get_type

    print("Normalizing parameters...")

    norm_dir = kwargs[str('path')] if 'path' in kwargs else \
        os.path.join(config.get('paths', 'log_dir_of_current_run'),
                     'normalization')

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
            len(x_norm) * np.array(layer.output_shape[1:], dtype='int64').prod() * 32 /
            (8 * 1e9) for layer in model.layers if len(layer.weights) > 0]
        size_str = ['{:.2f}'.format(s) for s in sizes]
        print("INFO: Need {} GB for layer activations.\n".format(size_str) +
              "May have to reduce size of data set used for normalization.")
        scale_facs = OrderedDict({model.layers[0].name: 1})
    else:
        import warnings
        warnings.warn("Scale factors or normalization data set could not be "
                      "loaded. Proceeding without normalization.",
                      RuntimeWarning)
        return

    batch_size = config.getint('simulation', 'batch_size')
    network_mode = config.getint('normalization', 'network_mode')

    # If scale factors have not been computed in a previous run, do so now.
    if len(scale_facs) == 1:
        i = 0
        sparsity = []
        for layer in model.layers:
            # Skip if layer has no parameters
            if len(layer.weights) == 0:
                continue

            activations = try_reload_activations(layer, model, x_norm,
                                                 batch_size, activ_dir)
            nonzero_activations = activations[np.nonzero(activations)]
            sparsity.append(1 - nonzero_activations.size / activations.size)
            del activations
            perc = get_percentile(config, i)
            scale_facs[layer.name] = get_scale_fac(nonzero_activations, perc)
            print("Scale factor: {:.2f}.".format(scale_facs[layer.name]))
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
            #               "half of the activations are negative, which "
            #               "could reduce accuracy. Consider setting "
            #               "settings['softmax_to_relu'] = False.")
            #         settings['softmax_to_relu'] = False
            i += 1
        # Write scale factors to disk
        filepath = os.path.join(norm_dir, config.get('normalization',
                                                     'percentile') + '.json')
        from snntoolbox.utils.utils import confirm_overwrite
        if config.get('output', 'overwrite') or confirm_overwrite(filepath):
            with open(filepath, str('w')) as f:
                json.dump(scale_facs, f)
        np.savez_compressed(os.path.join(norm_dir, 'activations', 'sparsity'),
                            sparsity=sparsity)

    # detect whether there is the CONV layer existing in the identity branch of the current resblock. 1 denotes no, 2 denotes existing. CONV layer here
    # is used to keep the channel the same, and prepare for point-wise addition in the Add layers. Here we call it channel-adaptive shortcut.
    test_idx = 0
    res_block = []
    res_block.append(1)
    for layer in model.layers:
        if len(layer.weights) == 0:
            continue
        inbound_identity_test = get_inbound_layers(layer)
        if get_type(inbound_identity_test[0]) == 'Add':
            test_idx += 1
        else:
            if test_idx != 0:
                res_block.append(test_idx)
            test_idx = 0
    res_block.append(0)
    res_block = np.array(res_block)

    # Apply scale factors to normalize the parameters.
    s_para_idx = 0
    i = 0
    ii = 0
    # compensation factors
    gamma = 1.04  #(1.0 ~ 1.1)
    beta = 1.21   #(1.0 ~ 1.3)
    for layer in model.layers:
        # Skip if layer has no parameters
        layer_type = get_type(layer)
        if layer_type == 'Add':
            i += 1

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
            scale_fac = 1
            print("Using scale factor {:.2f} for softmax layer.".format(
                scale_fac))
        else:
            scale_fac = scale_facs[layer.name]

        inbound = get_inbound_layers_with_params(layer)
        if len(inbound) == 0:  # Input layer
            parameters_norm = [
                parameters[0] * scale_facs[model.layers[0].name] / scale_fac,
                parameters[1] / scale_fac]
        elif len(inbound) == 1:
            inbound = get_inbound_layers_with_params(layer)
            outbound = get_outbound_layers(layer)
            outbound_type = get_type(outbound[0])
            # change the scaling factor of the current layer when the junction node occurs.
            if outbound_type == 'Add':
                if res_block[i] == 1:  # CONV layer in the non-identity branch of the resblock, connected to the Add layers
                    scale_fac1 = scale_facs[layer.name]
                    index = model.layers.index(layer)
                    if i == 0:
                        scale_fac2 = scale_facs[model.layers[index - 2].name]  # The first CONV layer in resblock, passing no Add layer before.
                    else:
                        scale_fac2 = scale_facs[model.layers[index - 3].name]
                    scale_fac = np.sqrt(scale_fac1 * scale_fac2 / gamma)
                    scale_facs[layer.name] = scale_fac
                if res_block[i] == 2:  # CONV layer in the non-identity branch of the resblock
                    scale_facs[layer.name] = scale_fac_rec
                    scale_fac = scale_fac_rec  # scale_fac_rec is calculated before in the CONV layer of the identity branch.
            parameters_norm = [
                parameters[0] * scale_facs[inbound[0].name] / scale_fac,
                parameters[1] / scale_fac]
        else:
            # determine the s_para_idx according to the res_block
            if res_block[i] == 1 and res_block[i - 1] != 2:
                s_para_idx = s_para_idx
            elif res_block[i] == 1 and res_block[i - 1] == 2:
                s_para_idx = 0
            elif res_block[i] == 2 and res_block[i - 1] != 2:
                ii += 1
                if ii % 2 == 1:
                    s_para_idx = s_para_idx
                else:
                    s_para_idx -= 1
            elif res_block[i] == 2 and res_block[i - 1] == 2:
                s_para_idx = 0
            elif res_block[i] == 0 and res_block[i - 1] == 2:
                s_para_idx = 0
            elif res_block[i] == 0 and res_block[i - 1] != 2:
                s_para_idx = s_para_idx

            # change the current scaling factor.
            # get the scale_fac used to be divided of the CONV layer in the identity branch in the resblock with channel-adaptive shortcut.
            if res_block[i] == 2:
                if parameters[0].ndim == 4:
                    outbound = get_outbound_layers(layer)
                    outbound_type = get_type(outbound[0])
                if outbound_type == 'Add':
                    scale_fac1 = scale_facs[layer.name]
                    index = model.layers.index(layer)
                    scale_fac2 = scale_facs[model.layers[index + 1].name]
                    scale_fac = np.sqrt(scale_fac1 * scale_fac2 * beta)  # beta is the compensation factor.
                    scale_fac_rec = scale_fac  # record the scale_fac, which will be used in the next CONV layer located at the non-identity branch.
                    scale_facs[layer.name] = scale_fac
                    # scale_facs[model.layers[index + 1].name] = scale_fac

            # get the scale_fac of the previous layer.
            if res_block[i] == 1 and res_block[i - 1] == 2:
                scale_facs_mean = scale_fac_rec
            else:
                scale_facs_mean = scale_facs[inbound[s_para_idx + 1].name]

            scale_facs[inbound[s_para_idx].name] = scale_facs_mean
            scale_facs[inbound[s_para_idx + 1].name] = scale_facs_mean

            if parameters[0].ndim == 4:
                if network_mode == 0:
                    # In conv layers, just need to split up along channel dim.
                    offset = 0  # Index offset at input filter dimension
                    for inb in inbound:
                        f_out = inb.filters  # Num output features of inbound layer
                        f_in = range(offset, offset + f_out)
                        parameters_norm[0][:, :, f_in, :] *= \
                            scale_facs[inb.name] / scale_fac
                        offset += f_out
                elif network_mode == 1:
                    parameters_norm = [
                        parameters[0] * scale_facs[inbound[s_para_idx + 1].name] / scale_fac,
                        parameters[1] / scale_fac]
            else:
                parameters_norm = [
                    parameters[0] * scale_facs[inbound[s_para_idx + 1].name] / scale_fac,
                    parameters[1] / scale_fac]

            s_para_idx += 1

        # Check if the layer happens to be Sparse
        # if the layer is sparse, add the mask to the list of parameters
        if len(parameters) == 3:
            parameters_norm.append(parameters[-1])
        # Update model with modified parameters
        layer.set_weights(parameters_norm)

    # Plot distributions of weights and activations before and after norm.
    if 'normalization_activations' in eval(config.get('output', 'plot_vars')):
        from snntoolbox.simulation.plotting import plot_hist
        from snntoolbox.simulation.plotting import plot_max_activ_hist

        # All layers in one plot. Assumes model.get_weights() returns
        # [w, b, w, b, ...].
        # from snntoolbox.simulation.plotting import plot_weight_distribution
        # plot_weight_distribution(norm_dir, model)

        print("Plotting distributions of weights and activations before and "
              "after normalizing...")

        # Load original parsed model to get parameters before normalization
        weights = np.load(os.path.join(activ_dir, 'weights.npz'))
        for idx, layer in enumerate(model.layers):
            # Skip if layer has no parameters
            if len(layer.weights) == 0:
                continue

            label = str(idx) + layer.__class__.__name__ \
                if config.getboolean('output', 'use_simple_labels') \
                else layer.name
            parameters = weights[layer.name]
            parameters_norm = layer.get_weights()[0]
            weight_dict = {'weights': parameters.flatten(),
                           'weights_norm': parameters_norm.flatten()}
            plot_hist(weight_dict, 'Weight', label, norm_dir)

            # Load activations of model before normalization
            activations = try_reload_activations(layer, model, x_norm,
                                                 batch_size, activ_dir)

            if activations is None or x_norm is None:
                continue

            # Compute activations with modified parameters
            nonzero_activations = activations[np.nonzero(activations)]
            activations_norm = get_activations_layer(model.input, layer.output,
                                                     x_norm, batch_size)
            activation_dict = {'Activations': nonzero_activations,
                               'Activations_norm':
                               activations_norm[np.nonzero(activations_norm)]}
            scale_fac = scale_facs[layer.name]
            plot_hist(activation_dict, 'Activation', label, norm_dir,
                      scale_fac)
            ax = tuple(np.arange(len(layer.output_shape))[1:])
            plot_max_activ_hist(
                {'Activations_max': np.max(activations, axis=ax)},
                'Maximum Activation', label, norm_dir, scale_fac)
    print('')


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

    return np.percentile(activations, percentile) if activations.size else 1


def get_percentile(config, layer_idx=None):
    """Get percentile at which to draw the maximum activation of a layer.

    Parameters
    ----------

    config: configparser.ConfigParser
        Settings.

    layer_idx: Optional[int]
        Layer index.

    Returns
    -------

    : int
        Percentile.

    """

    perc = config.getfloat('normalization', 'percentile')

    if config.getboolean('normalization', 'normalization_schedule'):
        assert layer_idx, "Layer index needed for normalization schedule."
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

    activations: ndarray
        The activations of cells in a specific layer. Has the same shape as
        ``layer_out``.
    """

    if batch_size is None:
        batch_size = 10

    if len(x) % batch_size != 0:
        x = x[: -(len(x) % batch_size)]

    return keras.models.Model(layer_in, layer_out).predict(x, batch_size)


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
        # Todo: This list should be replaced by
        #       ``not in eval(config.get('restrictions', 'spiking_layers')``
        if layer.__class__.__name__ in ['Input', 'InputLayer', 'Flatten',
                                        'Concatenate', 'ZeroPadding2D',
                                        'Reshape', 'Add']: # qinyu changes
            continue
        activations = keras.models.Model(
            ann.input, layer.output).predict_on_batch(x_batch)
        activations_batch.append((activations, layer.name))
    return activations_batch


def try_reload_activations(layer, model, x_norm, batch_size, activ_dir):
    try:
        activations = np.load(os.path.join(activ_dir,
                                           layer.name + '.npz'))['arr_0']
    except IOError:
        if x_norm is None:
            return

        print("Calculating activations of layer {} ...".format(layer.name))
        activations = get_activations_layer(model.input, layer.output, x_norm,
                                            batch_size)
        print("Writing activations to disk...")
        np.savez_compressed(os.path.join(activ_dir, layer.name), activations)
    else:
        print("Loading activations stored during a previous run.")
    return np.array(activations)
