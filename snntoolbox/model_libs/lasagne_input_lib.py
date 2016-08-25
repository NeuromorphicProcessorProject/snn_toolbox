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
import numpy as np

from snntoolbox.config import settings, spiking_layers
from snntoolbox.io_utils.load import load_parameters
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

        - parameters (array): The weights and biases connecting this layer with
          the previous.

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
    parameters = lasagne.layers.get_all_param_values(model)
    input_shape = lasagne_layers[0].shape

    layers = []
    labels = []
    layer_idx_map = []
    parameters_idx = 0
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
            layer_type = 'BatchNormalization'
        elif name == 'NonlinearityLayer':
            layer_type = 'Activation'
        else:
            layer_type = layer.__class__.__name__
        attributes = {'layer_num': idx,
                      'layer_type': layer_type,
                      'output_shape': layer.output_shape}

        if layer_type == 'BatchNormalization':
            bn_parameters = parameters[parameters_idx: parameters_idx + 4]
            for k in [layer_num - i for i in range(1, 3)]:
                prev_layer = lasagne_layers[k]
                label = prev_layer.__class__.__name__
                if prev_layer.get_params() != []:
                    break
            wb = parameters[parameters_idx - 2: parameters_idx]
            print("Absorbing batch-normalization parameters into " +
                  "parameters of layer {}, {}.".format(k, label))
            parameters_norm = absorb_bn(wb[0], wb[1],
                                        bn_parameters[1], bn_parameters[0],
                                        bn_parameters[2], 1/bn_parameters[3],
                                        layer.epsilon)
            # Remove Batch-normalization layer by setting gamma=1, beta=1,
            # mean=0, std=1
            zeros = np.zeros_like(bn_parameters[0])
            ones = np.ones_like(bn_parameters[0])
            layer.gamma.set_value(ones)
            layer.beta.set_value(zeros)
            layer.mean.set_value(zeros)
            layer.inv_std.set_value(ones)
            # Replace parameters of preceding Conv or FC layer by parameters
            # that include the batch-normalization transformation.
            prev_layer.W.set_value(parameters_norm[0])
            prev_layer.b.set_value(parameters_norm[1])
            layers[-1]['parameters'] = parameters_norm
            parameters_idx += 4
            continue

        if layer_type not in spiking_layers:
            continue

        # Insert Flatten layer
        prev_layer_output_shape = lasagne_layers[layer_num-1].output_shape
        if len(attributes['output_shape']) < len(prev_layer_output_shape) and \
                layer_type != 'Flatten':
            print("Inserting Flatten layer")
            flat_output_shape = [layers[-1]['output_shape'][0],
                                 np.prod(layers[-1]['output_shape'][1:])]
            attributes2 = {'layer_num': idx,
                           'layer_type': 'Flatten',
                           'output_shape': flat_output_shape}
            # Append layer label
            num_str = str(idx) if idx > 9 else '0' + str(idx)
            shape_string = '_{}'.format(attributes2['output_shape'][1])
            labels.append(num_str + attributes2['layer_type'] + shape_string)
            attributes2.update({'label': labels[-1]})
            layers.append(attributes2)
            layer_idx_map.append(layer_num)
            idx += 1

        # Append layer label
        if len(attributes['output_shape']) == 2:
            shape_string = '_{}'.format(attributes['output_shape'][1])
        else:
            shape_string = '_{}x{}x{}'.format(attributes['output_shape'][1],
                                              attributes['output_shape'][2],
                                              attributes['output_shape'][3])
        num_str = str(idx) if idx > 9 else '0' + str(idx)
        labels.append(num_str + layer_type + shape_string)
        attributes.update({'label': labels[-1]})

        if layer_type in {'Dense', 'Convolution2D'}:
            wb = parameters[parameters_idx: parameters_idx + 2]
            parameters_idx += 2  # For weights and biases
            # Get type of nonlinearity if the activation is directly in the
            # Dense / Conv layer:
            activation = layer.nonlinearity.__name__
            # Otherwise, search for the activation layer:
            for k in range(layer_num+1, min(layer_num+4, len(lasagne_layers))):
                next_layer = lasagne_layers[k]
                if next_layer.__class__.__name__ == 'NonlinearityLayer':
                    nonlinearity = next_layer.nonlinearity.__name__
                    if nonlinearity == 'rectify':
                        activation = 'relu'
                        layer.nonlinearity = next_layer.nonlinearity
                    elif nonlinearity == 'softmax':
                        activation = 'softmax'
                        layer.nonlinearity = next_layer.nonlinearity
                    elif nonlinearity == 'binary_tanh_unit':
                        activation = 'softsign'
                        layer.nonlinearity = next_layer.nonlinearity
                    else:
                        activation = 'linear'
                        layer.nonlinearity = lasagne.nonlinearities.linear
                    break
            attributes.update({'parameters': wb,
                               'activation': activation,
                               'get_activ': get_activ_fn_for_layer(model,
                                                                   layer_num)})

        if layer_type == 'Convolution2D':
            border_mode = border_mode_string(layer.pad, layer.filter_size)
            attributes.update({'input_shape': layer.input_shape,
                               'nb_filter': layer.num_filters,
                               'nb_col': layer.filter_size[1],
                               'nb_row': layer.filter_size[0],
                               'border_mode': border_mode,
                               'subsample': layer.stride,
                               'filter_flip': layer.flip_filters})

        if layer_type in {'MaxPooling2D', 'AveragePooling2D'}:
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

    return {'input_shape': input_shape, 'layers': layers, 'labels': labels,
            'layer_idx_map': layer_idx_map}


def get_activ_fn_for_layer(model, i):
    layers = lasagne.layers.get_all_layers(model)
    f = theano.function(
        [layers[0].input_var, theano.In(theano.tensor.scalar(), value=0)],
        lasagne.layers.get_output(layers[i], layers[0].input_var),
        allow_input_downcast=True, on_unused_input='ignore')
    return lambda x: f(x).astype('float16', copy=False)


def model_from_py(path=None, filename=None):
    if path is None:
        path = settings['path']
    if filename is None:
        filename = settings['filename']

    mod = import_script(path, filename)
    model, train_fn, val_fn = mod.build_network()
    params = load_parameters(os.path.join(path, filename + '.h5'))
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


def save_parameters(parameters, path):
    """
    Dump all layer parameters to a HDF5 file.

    """

    import h5py

    f = h5py.File(path, 'w')

    for i, p in enumerate(parameters):
        idx = '0' + str(i) if i < 10 else str(i)
        f.create_dataset('param_' + idx, data=p, dtype=p.dtype)
    f.flush()
    f.close()


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
