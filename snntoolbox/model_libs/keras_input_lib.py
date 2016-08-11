# -*- coding: utf-8 -*-
"""
Methods to parse an input model written in Keras and prepare it for further
processing in the SNN toolbox.

The idea is to make all further steps in the conversion/simulation pipeline
independent of the original model format. Therefore, when a developer adds a
new input model library (e.g. Caffe) to the toolbox, the following methods must
be implemented and satisfy the return requirements specified in their
respective docstrings:

    - extract
    - evaluate
    - load_ann

Created on Thu May 19 08:21:05 2016

@author: rbodo
"""

import os
import theano
from keras import backend as K
from snntoolbox.config import settings, bn_layers
from snntoolbox.model_libs.common import absorb_bn, import_script


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

    input_shape = model.input_shape

    layers = []
    labels = []
    layer_idx_map = []
    for (layer_num, layer) in enumerate(model.layers):
        if 'BatchNormalization' in layer.__class__.__name__:
            continue
        attributes = {'layer_num': layer_num,
                      'layer_type': layer.__class__.__name__,
                      'output_shape': layer.output_shape}

        # Append layer label
        if len(attributes['output_shape']) == 2:
            shape_string = '_{}'.format(attributes['output_shape'][1])
        else:
            shape_string = '_{}x{}x{}'.format(attributes['output_shape'][1],
                                              attributes['output_shape'][2],
                                              attributes['output_shape'][3])
        num_str = str(layer_num) if layer_num > 9 else '0' + str(layer_num)
        labels.append(num_str + attributes['layer_type'] + shape_string)
        attributes.update({'label': labels[-1]})

        next_layer = model.layers[layer_num + 1] \
            if layer_num + 1 < len(model.layers) else None
        next_layer_name = next_layer.__class__.__name__ if next_layer else None
        if next_layer_name == 'BatchNormalization' and \
                attributes['layer_type'] not in bn_layers:
            raise NotImplementedError(
                "A batchnormalization layer must follow a layer of type " +
                "{}, not {}.".format(bn_layers, attributes['layer_type']))

        if attributes['layer_type'] in {'Dense', 'Convolution2D'}:
            parameters = layer.get_weights()
            if next_layer_name == 'BatchNormalization':
                bn_parameters = next_layer.get_weights()
                # W, b, beta, gamma, mean, std, epsilon
                parameters_norm = absorb_bn(parameters[0], parameters[1],
                                            bn_parameters[0], bn_parameters[1],
                                            bn_parameters[2], bn_parameters[3],
                                            next_layer.epsilon)
                if len(parameters[0].shape) == 4:
                    print("epsilon: {}".format(next_layer.epsilon))
                    print("w: {}".format(parameters[0][0, 0, 0, 0]))
                    print("gamma: {}".format(bn_parameters[1][0]))
                    print("std: {}".format(bn_parameters[3][0]))
                    print("w_norm: {}".format(parameters_norm[0][0, 0, 0, 0]))
                    print("w_calc: {}".format(parameters[0][0, 0, 0, 0] * bn_parameters[1][0] / bn_parameters[3][0]))
                set_layer_params(model, parameters_norm, layer_num)
                parameters = parameters_norm
            if next_layer_name == 'Activation':
                attributes.update({'activation':
                                   next_layer.get_config()['activation']})
            attributes.update({'parameters': parameters})

        if attributes['layer_type'] == 'Convolution2D':
            attributes.update({'input_shape': layer.input_shape,
                               'nb_filter': layer.nb_filter,
                               'nb_col': layer.nb_col,
                               'nb_row': layer.nb_row,
                               'border_mode': layer.border_mode,
                               'subsample': layer.subsample,
                               'filter_flip': True})

        if attributes['layer_type'] in {'MaxPooling2D', 'AveragePooling2D'}:
            attributes.update({'input_shape': layer.input_shape,
                               'pool_size': layer.pool_size,
                               'strides': layer.strides,
                               'border_mode': layer.border_mode})

        if attributes['layer_type'] in {'Activation', 'AveragePooling2D',
                                        'MaxPooling2D'}:
            attributes.update({'get_activ': get_activ_fn_for_layer(model,
                                                                   layer_num)})
        layers.append(attributes)
        layer_idx_map.append(layer_num)

    return {'input_shape': input_shape, 'layers': layers, 'labels': labels,
            'layer_idx_map': layer_idx_map}


def get_activ_fn_for_layer(model, i):
    f = theano.function(
        [model.layers[0].input, theano.In(K.learning_phase(), value=0)],
        model.layers[i].output, allow_input_downcast=True,
        on_unused_input='ignore')
    return lambda x: f(x).astype('float16', copy=False)


def model_from_py(path=None, filename=None):
    if path is None:
        path = settings['path']
    if filename is None:
        filename = settings['filename']
    mod = import_script(path, filename)
    return {'model': mod.build_network()}


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

    if path is None:
        path = settings['path']
    if filename is None:
        filename = settings['filename']

    from keras import models
    model = models.model_from_json(open(os.path.join(
        path, filename + '.json')).read())
    model.load_weights(os.path.join(path, filename + '.h5'))
    # Todo: Allow user to specify loss function here (optimizer is not
    # relevant as we do not train any more). Unfortunately, Keras does not
    # save these parameters. They can be obtained from the compiled model
    # by calling 'model.loss' and 'model.optimizer'.
    model.compile(loss='categorical_crossentropy', optimizer='sgd',
                  metrics=['accuracy'])
    return {'model': model, 'val_fn': model.evaluate}


def evaluate(val_fn, X_test, Y_test):
    """Evaluate the original ANN."""
    return val_fn(X_test, Y_test)


def set_layer_params(model, params, i):
    """Set ``params`` of layer ``i`` of a given ``model``."""
    model.layers[i].set_weights(params)
