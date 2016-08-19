# -*- coding: utf-8 -*-
"""
Methods to parse an input model written in caffe and prepare it for further
processing in the SNN toolbox.

The idea is to make all further steps in the conversion/simulation pipeline
independent of the original model format. Therefore, when a developer adds a
new input model library (e.g. Torch) to the toolbox, the following methods must
be implemented and satisfy the return requirements specified in their
respective docstrings:

    - extract
    - evaluate
    - load_ann

Created on Thu Jun  9 08:11:09 2016

@author: rbodo
"""

import os
import caffe
import numpy as np
from snntoolbox.config import settings, bn_layers
from snntoolbox.model_libs.common import absorb_bn, border_mode_string

caffe.set_mode_gpu()


def extract(model):
    """
    Extract the essential information about a neural network.

    This method serves to abstract the conversion process of a network from the
    language the input model was built in (e.g. Keras or Lasagne).

    To extend the toolbox by another input format (e.g. Torch), this method has
    to be implemented for the respective model library.

    Parameters
    ----------

    model: dict
        A dictionary of objects that constitute the input model. Contains at
        least the keys:
        - 'model': A model instance of the network in the respective
          ``model_lib``.
        - 'model_protobuf': caffe.proto.caffe_pb2.NetParameter protocol buffer.
          The result of reading out the network specification from the prototxt
          file.
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

    model_protobuf = model['model_protobuf']
    model = model['model']

    input_shape = list(model_protobuf.input_dim)
    if input_shape == []:
        input_shape = list(model_protobuf.layer[0].input_param.shape[0].dim)

    global layer_keys
    layer_keys = [layer.name for layer in model_protobuf.layer]
    layers = []
    labels = []
    layer_idx_map = []
    idx = 0
    for (layer_num, layer) in enumerate(model_protobuf.layer):

        # Convert Caffe layer names to our 'standard' names.
        name = layer.type
        if name == 'InnerProduct':
            layer_type = 'Dense'
        elif name == 'Convolution':
            layer_type = 'Convolution2D'
        elif name == 'Pooling':
            pooling = layer.pooling_param.PoolMethod.DESCRIPTOR.values[0].name
            if pooling == 'MAX':
                layer_type = 'MaxPooling2D'
            else:
                layer_type = 'AveragePooling2D'
        elif name in {'ReLU', 'Softmax'}:
            layer_type = 'Activation'
        elif name == 'Input':
            continue
        else:
            layer_type = name

        layer_key = layer.name
        if layer_key not in model.blobs:
            # Assume shape is unchanged if not explicitly given
            output_shape = layers[-1]['output_shape']
        else:
            output_shape = list(model.blobs[layer_key].shape)
        attributes = {'layer_num': idx,
                      'layer_type': layer_type,
                      'output_shape': output_shape}

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

        next_layer = model.layers[layer_num + 1] \
            if layer_num + 1 < len(model.layers) else None
        next_layer_type = next_layer.type if next_layer else None
        if next_layer_type == 'BatchNormalization' and \
                attributes['layer_type'] not in bn_layers:
            raise NotImplementedError(
                "A batchnormalization layer must follow a layer of type " +
                "{}, not {}.".format(bn_layers, attributes['layer_type']))

        if attributes['layer_type'] in {'Dense', 'Convolution2D'}:
            wb = [model.params[layer_key][0].data,
                  model.params[layer_key][1].data]
            if next_layer_type == 'BatchNormalization':
                parameters = [next_layer.blobs[0].data,
                              next_layer.blobs[1].data]
                # W, b, gamma, beta, mean, std, epsilon
                wb = absorb_bn(wb[0], wb[1], parameters[0], parameters[1],
                               parameters[2], parameters[3],
                               next_layer.epsilon)
            if next_layer_type in {'ReLU', 'Softmax'}:
                a = 'softmax' if next_layer_type == 'Softmax' else 'relu'
                attributes.update({'activation': a})
            if attributes['layer_type'] == 'Dense':
                wb[0] = np.transpose(wb[0])
            attributes.update({'parameters': wb})

        if attributes['layer_type'] == 'Convolution2D':
            p = layer.convolution_param
            # Take maximum here because sometimes not not all fields are set
            # (e.g. kernel_h == 0 even though kernel_size == [3])
            filter_size = [max(p.kernel_w, p.kernel_size[0]),
                           max(p.kernel_h, p.kernel_size[-1])]
            pad = (p.pad_w, p.pad_h)
            border_mode = border_mode_string(pad, filter_size)
            inp_shape = input_shape if len(layers) == 0 else \
                layers[-1]['output_shape']
            attributes.update({'input_shape': inp_shape,
                               'nb_filter': p.num_output,
                               'nb_col': filter_size[0],
                               'nb_row': filter_size[1],
                               'border_mode': border_mode,
                               'subsample': (p.stride[0], p.stride[0]),
                               'filter_flip': False})  # p.filter_flip

        if attributes['layer_type'] in {'MaxPooling2D', 'AveragePooling2D'}:
            p = layer.pooling_param
            # Take maximum here because sometimes not not all fields are set
            # (e.g. kernel_h == 0 even though kernel_size == 2)
            pool_size = [max(p.kernel_w, p.kernel_size),
                         max(p.kernel_h, p.kernel_size)]
            pad = (max(p.pad_w, p.pad), max(p.pad_h, p.pad))
            border_mode = border_mode_string(pad, pool_size)
            strides = [max(p.stride_w, p.stride), max(p.stride_h, p.stride)]
            inp_shape = input_shape if len(layers) == 0 else \
                layers[-1]['output_shape']
            attributes.update({'input_shape': inp_shape,
                               'pool_size': pool_size,
                               'strides': strides,
                               'border_mode': border_mode})

        if attributes['layer_type'] in {'Activation', 'AveragePooling2D',
                                        'MaxPooling2D'}:
            if attributes['layer_type'] == 'Activation' and \
                    layer.name != 'prob':
                layer_key = model_protobuf.layer[layer_num-1].name
            attributes.update({'get_activ': get_activ_fn_for_layer(model,
                                                                   layer_key)})
        layers.append(attributes)
        layer_idx_map.append(layer_key)
        idx += 1

        # Insert Flatten layer
        if layer_num + 1 == len(model_protobuf.layer):
            continue
        next_layer_key = model_protobuf.layer[layer_num+1].name
        if next_layer_key not in model.blobs:
            # Assume shape is unchanged if not explicitly given
            next_layer_output_shape = layers[-1]['output_shape']
        else:
            next_layer_output_shape = list(model.blobs[next_layer_key].shape)
        if len(attributes['output_shape']) > len(next_layer_output_shape):
            flat_output_shape = [layers[-1]['output_shape'][0],
                                 np.prod(layers[-1]['output_shape'][1:])]
            attributes = {'layer_num': idx,
                          'layer_type': 'Flatten',
                          'output_shape': flat_output_shape}
            # Append layer label
            num_str = str(idx) if idx > 9 else '0' + str(idx)
            shape_string = '_{}'.format(attributes['output_shape'][1])
            labels.append(num_str + attributes['layer_type'] + shape_string)
            attributes.update({'label': labels[-1]})
            layers.append(attributes)
            layer_idx_map.append('flat')
            idx += 1

    return {'input_shape': input_shape, 'layers': layers, 'labels': labels,
            'layer_idx_map': layer_idx_map}


def get_activ_fn_for_layer(model, layer_key):
    return lambda x: model.forward(data=x, blobs=[layer_key])[layer_key]


def load_ann(path=None, filename=None):
    """
    Load network from file.

    Parameters
    ----------

    path: string, optional
        Path to directory where to load model from. Defaults to
        ``settings['path']``.

    filename: string, optional
        Name of file to load model from. Defaults to ``settings['filename']``.

    Returns
    -------

    model: dict
        A dictionary of objects that constitute the input model. It must
        contain the following two keys:

        - 'model': Model instance of the network in the respective
          ``model_lib``.
        - 'val_fn': Theano function that allows evaluating the original model.

        For instance, if the input model was written using Keras, the 'model'-
        value would be an instance of ``keras.Model``, and 'val_fn' the
        ``keras.Model.evaluate`` method.

    """

    from google.protobuf import text_format

    if path is None:
        path = settings['path']
    if filename is None:
        filename = settings['filename']

    prototxt = os.path.join(path, filename + '.prototxt')
    caffemodel = os.path.join(path, filename + '.caffemodel')
    model = caffe.Net(prototxt, caffemodel, caffe.TEST)
    model_protobuf = caffe.proto.caffe_pb2.NetParameter()
    text_format.Merge(open(prototxt).read(), model_protobuf)
    return {'model': model, 'val_fn': model.forward_all,
            'model_protobuf': model_protobuf}


def evaluate(val_fn, X_test, Y_test):
    """Evaluate the original ANN."""
    guesses = np.argmax(val_fn(data=X_test)['prob'], axis=1)
    truth = np.argmax(Y_test, axis=1)
    accuracy = np.mean(guesses == truth)
    loss = -1
    return [loss, accuracy]


def set_layer_params(model, params, layer_key):
    """Set ``params`` of layer ``i`` of a given ``model``."""
    input_offset = 0 if layer_keys[0] == 'data' else 1
    i = layer_keys.index(layer_key) + input_offset
    params0 = np.transpose(params[0]) if 'ip' in layer_key else params[0]
    model.params[layer_key][0].data[...] = params0
    model.params[layer_key][1].data[...] = params[1]
    model.layers[i].blobs[0].data[...] = params0
    model.layers[i].blobs[1].data[...] = params[1]
