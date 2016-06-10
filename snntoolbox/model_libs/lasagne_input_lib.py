# -*- coding: utf-8 -*-
"""
Methods to parse an input model written in Lasagne and prepare it for further
processing in the SNN toolbox.

The idea is to make all further steps in the conversion/simulation pipeline
independent of the original model format. Therefore, when a developer adds a
new input model library (e.g. torch) to the toolbox, the following methods must
be implemented and satisfy the return requirements specified in their
respective docstrings:

    - extract
    - evaluate
    - load_ann

Created on Thu Jun  9 08:11:09 2016

@author: rbodo
"""

import os
import lasagne
import theano
from snntoolbox.config import settings, activation_layers, bn_layers
from snntoolbox.io.load import load_weights
from snntoolbox.model_libs.common import absorb_bn, import_script
from snntoolbox.model_libs.common import border_mode_string


def extract(model):
    """
    Extract the essential information about a neural network.

    This method serves to abstract the conversion process of a network from the
    language the input model was built in (e.g. Keras or Lasagne).

    To extend the toolbox by another input format (e.g. Caffe), this method has
    to be implemented for the respective model library.

    Parameters
    ----------

    model: dict
        A dictionary of objects that constitute the input model. Contains at
        least the key
        - 'model': A model instance of the network in the respective
          ``model_lib``.
        For instance, if the input model was written using Keras, the 'model'-
        value would be an instance of ``keras.Model``.

    Returns
    -------

    Dictionary containing the parsed network.

    input_shape: list
        The dimensions of the input sample
        [batch_size, n_chnls, n_rows, n_cols]. For instance, mnist would have
        input shape [Null, 1, 28, 28].

    layers: list
        List of all the layers of the network, where each layer contains a
        dictionary with keys

        - layer_num (int): Index of layer.
        - layer_type (string): Describing the type, e.g. `Dense`,
          `Convolution`, `Pool`.
        - output_shape (list): The output dimensions of the layer.

        In addition, `Dense` and `Convolution` layer types contain

        - weights (array): The weight parameters connecting this layer with the
          previous.

        `Convolution` layers contain further

        - nb_col (int): The x-dimension of filters.
        - nb_row (int): The y-dimension of filters.
        - border_mode (string): How to handle borders during convolution, e.g.
          `full`, `valid`, `same`.

        `Pooling` layers contain

        - pool_size (list): Specifies the subsampling factor in each dimension.
        - strides (list): The stepsize in each dimension during pooling.

        `Activation` layers (including Pooling) contain

        - get_activ: A Theano function computing the activations of a layer.

    labels: list
        The layer labels.

    layer_idx_map: list
        A list mapping the layer indices of the original network to the parsed
        network. (Not all layers of the original model are needed in the parsed
        model.) For instance: To get the layer index i of the original input
        ``model`` that corresponds to layer j of the parsed network ``layers``,
        one would use ``i = layer_idx_map[j]``.

    """

    model = model['model']

    lasagne_layers = lasagne.layers.get_all_layers(model)
    weights = lasagne.layers.get_all_param_values(model)
    input_shape = lasagne_layers[0].shape

    layers = []
    labels = []
    layer_idx_map = []
    weights_idx = 0
    idx = 0
    for (layer_num, layer) in enumerate(lasagne_layers):

        # Convert Lasagne layer names to our 'standard' names.
        name = layer.__class__.__name__
        if name == 'DenseLayer':
            layer_type = 'Dense'
        elif name in {'Conv2DLayer', 'Conv2DDNNLayer'}:
            layer_type = 'Convolution2D'
        elif name == 'MaxPool2DLayer':
            layer_type = 'MaxPooling2D'
        elif name in {'Pool2DLayer'}:
            layer_type = 'AveragePooling2D'
        elif name == 'DropoutLayer':
            layer_type = 'Dropout'
        elif name == 'FlattenLayer':
            layer_type = 'Flatten'
        elif name == 'BatchNormLayer':
            layer_type = 'BatchNorm'
        elif name == 'InputLayer':
            continue
        else:
            layer_type = layer.__class__.__name__
        attributes = {'layer_num': idx,
                      'layer_type': layer_type,
                      'output_shape': layer.output_shape}

        # Append layer label
        if len(attributes['output_shape']) == 2:
            shape_string = '_{}'.format(attributes['output_shape'][1])
        else:
            shape_string = '_{}x{}x{}'.format(attributes['output_shape'][1],
                                              attributes['output_shape'][2],
                                              attributes['output_shape'][3])
        num_str = str(idx) if idx > 9 else '0' + str(idx)
        labels.append(num_str + attributes['layer_type'] + shape_string)
        attributes.update({'label': labels[-1]})

        next_layer = lasagne_layers[layer_num + 1] \
            if layer_num + 1 < len(lasagne_layers) else None
        next_layer_name = next_layer.__class__.__name__ if next_layer else None
        if next_layer_name == 'BatchNormLayer' and \
                attributes['layer_type'] not in bn_layers:
            raise NotImplementedError(
                "A batchnormalization layer must follow a layer of type " +
                "{}, not {}.".format(bn_layers, attributes['layer_type']))

        if attributes['layer_type'] in {'Dense', 'Convolution2D'}:
            wb = weights[weights_idx: weights_idx + 2]
            weights_idx += 2  # For weights and biases
            if next_layer_name == 'BatchNormLayer':
                wb = absorb_bn(wb[0], wb[1],  # W, b
                               weights[weights_idx + 0],  # gamma
                               weights[weights_idx + 1],  # beta
                               weights[weights_idx + 2],  # mean
                               weights[weights_idx + 3],  # std
                               next_layer.epsilon)
                weights_idx += 4
            attributes.update({'weights': wb})

        if attributes['layer_type'] == 'Convolution2D':
            border_mode = border_mode_string(layer.pad, layer.filter_size)
            attributes.update({'input_shape': layer.input_shape,
                               'nb_filter': layer.num_filters,
                               'nb_col': layer.filter_size[1],
                               'nb_row': layer.filter_size[0],
                               'border_mode': border_mode})

        if attributes['layer_type'] in {'MaxPooling2D', 'AveragePooling2D'}:
            border_mode = border_mode_string(layer.pad, layer.pool_size)
            attributes.update({'input_shape': layer.input_shape,
                               'pool_size': layer.pool_size,
                               'strides': layer.stride,
                               'border_mode': border_mode,
                               'get_activ': get_activ_fn_for_layer(model,
                                                                   layer_num)})

        # Append layer
        layers.append(attributes)
        layer_idx_map.append(layer_num)
        idx += 1

        # Add activation layer after Dense, Conv, etc.
        # Probably need to adapt this in case there is a BatchNorm layer.
        if attributes['layer_type'] in activation_layers:
            attributes = {'layer_num': idx,
                          'layer_type': 'Activation',
                          'output_shape': layer.output_shape}
            # Append layer label
            num_str = str(idx) if idx > 9 else '0' + str(idx)
            labels.append(num_str + attributes['layer_type'] + shape_string)
            attributes.update({'label': labels[-1]})
            attributes.update({'get_activ': get_activ_fn_for_layer(model,
                                                                   layer_num)})
            layers.append(attributes)
            layer_idx_map.append(layer_num)
            idx += 1

    return {'input_shape': input_shape, 'layers': layers, 'labels': labels,
            'layer_idx_map': layer_idx_map}


def get_activ_fn_for_layer(model, i):
    layers = lasagne.layers.get_all_layers(model)
    return theano.function(
        [layers[0].input_var, theano.In(theano.tensor.scalar(), value=0)],
        lasagne.layers.get_output(layers[i], layers[0].input_var),
        allow_input_downcast=True, on_unused_input='ignore')


def model_from_py(path=None, filename=None):
    if path is None:
        path = settings['path']
    if filename is None:
        filename = settings['filename']

    mod = import_script(path, filename)
    model, train_fn, val_fn = mod.build_network()
    params = load_weights(os.path.join(path, filename + '.h5'))
    lasagne.layers.set_all_param_values(model, params)

    return {'model': model, 'val_fn': val_fn}


def load_ann(path=None, filename=None):
    """
    Load network from file.

    Parameters
    ----------

        path: string, optional
            Path to directory where to load model from. Defaults to
            ``settings['path']``.

        filename: string, optional
            Name of file to load model from. Defaults to
            ``settings['filename']``.

    Returns
    -------

    model: dict
        A dictionary of objects that constitute the input model. It must
        contain the following two keys:

        - 'model': Model instance of the network in the respective
          ``model_lib``.
        - 'val_fn': Theano function that allows evaluating the original
          model.

        For instance, if the input model was written using Keras, the
        'model'-value would be an instance of ``keras.Model``, and
        'val_fn' the ``keras.Model.evaluate`` method.

    """

    if path is None:
        path = settings['path']
    if filename is None:
        filename = settings['filename']

    return model_from_py(path, filename)


def evaluate(val_fn, X_test, Y_test):
    """
    Test a lasagne model batchwise on the whole dataset.

    """

    err = 0
    loss = 0
    batch_size = settings['batch_size']
    batches = int(len(X_test) / batch_size)

    for i in range(batches):
        new_loss, new_err = val_fn(X_test[i*batch_size: (i+1)*batch_size],
                                   Y_test[i*batch_size: (i+1)*batch_size])
        err += new_err
        loss += new_loss

    err /= batches
    loss /= batches

    return loss, 1 - err  # Convert error into accuracy here.


def set_layer_params(model, params, i):
    """Set ``params`` of layer ``i`` of a given ``model``."""
    layers = lasagne.layers.get_all_layers(model)
    layers[i].W.set_value(params[0])
    layers[i].b.set_value(params[1])
