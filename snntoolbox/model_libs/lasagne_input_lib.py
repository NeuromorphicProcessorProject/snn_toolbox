# -*- coding: utf-8 -*-
"""
Created on Thu May 19 08:26:00 2016

@author: rbodo
"""

import os
import lasagne
import theano
from snntoolbox.config import settings, activation_layers, bn_layers
from snntoolbox.io.load import load_weights
from snntoolbox.model_libs.common import absorb_bn, import_script


def extract(model):
    lasagne_layers = lasagne.layers.get_all_layers(model)
    weights = lasagne.layers.get_all_param_values(model)
    input_shape = lasagne_layers[0].shape

    layers = []
    labels = []
    layer_idx_map = []
    weights_idx = 0
    idx = 0
    for (layer_num, layer) in enumerate(lasagne_layers):
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
            fs = layer.filter_size
            if layer.pad == (0, 0):
                border_mode = 'valid'
            elif layer.pad == (fs[0] // 2, fs[1] // 2):
                border_mode = 'same'
            elif layer.pad == (fs[0] - 1, fs[1] - 1):
                border_mode = 'full'
            else:
                raise NotImplementedError("Padding {} ".format(layer.pad) +
                                          "could not be interpreted as any " +
                                          "of the supported border modes " +
                                          "'valid', 'same' or 'full'.")
            attributes.update({'input_shape': layer.input_shape,
                               'nb_filter': layer.num_filters,
                               'nb_col': fs[1],
                               'nb_row': fs[0],
                               'border_mode': border_mode})

        elif attributes['layer_type'] in {'MaxPooling2D', 'AveragePooling2D'}:
            ps = layer.pool_size
            if layer.pad == (0, 0):
                border_mode = 'valid'
            elif layer.pad == (ps[0] // 2, ps[1] // 2):
                border_mode = 'same'
            elif layer.pad == (ps[0] - 1, ps[1] - 1):
                border_mode = 'full'
            else:
                raise NotImplementedError("Padding {} ".format(layer.pad) +
                                          "could not be interpreted as any " +
                                          "of the supported border modes " +
                                          "'valid', 'same' or 'full'.")
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


def model_from_py(filename):
    mod = import_script(filename)
    model, train_fn, val_fn = mod.build_network()
    params = load_weights(os.path.join(settings['path'], filename + '.h5'))
    lasagne.layers.set_all_param_values(model, params)
    return {'model': model, 'val_fn': val_fn}


def load_ann(filename):
    return model_from_py(filename)


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
    layers = lasagne.layers.get_all_layers(model)
    layers[i].W.set_value(params[0])
    layers[i].b.set_value(params[1])
