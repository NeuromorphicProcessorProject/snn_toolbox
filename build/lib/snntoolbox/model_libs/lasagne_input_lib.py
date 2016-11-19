# -*- coding: utf-8 -*-
"""
Methods to extract an input model written in Lasagne and prepare it for further
processing in the SNN toolbox.

The idea is to make all further steps in the conversion/simulation pipeline
independent of the original model format. Therefore, when a developer adds a
new input model library (e.g. torch) to the toolbox, the following methods must
be implemented and satisfy the return requirements specified in their
respective docstrings:

    - extract
    - load_ann
    - evaluate

Created on Thu Jun  9 08:11:09 2016

@author: rbodo
"""

import os
from typing import Optional

import lasagne
import numpy as np
import theano
from snntoolbox.config import settings, spiking_layers
from snntoolbox.io_utils.common import load_parameters
from snntoolbox.model_libs.common import border_mode_string
from snntoolbox.model_libs.common import import_script

layer_dict = {'DenseLayer': 'Dense',
              'Conv2DLayer': 'Convolution2D',
              'Conv2DDNNLayer': 'Convolution2D',
              'MaxPool2DLayer': 'MaxPooling2D',
              'Pool2DLayer': 'AveragePooling2D',
              'DropoutLayer': 'Dropout',
              'FlattenLayer': 'Flatten',
              'BatchNormLayer': 'BatchNormalization',
              'NonlinearityLayer': 'Activation'}


activation_dict = {'rectify': 'relu',
                   'softmax': 'softmax',
                   'binary_tanh_unit': 'binary_tanh',
                   'binary_sigmoid_unit': 'binary_sigmoid',
                   'linear': 'linear'}


def extract(model):
    """Extract the essential information about a neural network.

    This method serves to abstract the conversion process of a network from the
    language the input model was built in (e.g. Keras or Lasagne).

    To extend the toolbox by another input format (e.g. Caffe), this method has
    to be implemented for the respective model library.

    Implementation details:
    The methods iterates over all layers of the input model and writes the
    layer specifications and parameters into a dictionary. The keys are chosen
    in accordance with Keras layer attributes to facilitate instantiation of a
    new, parsed Keras model (done in a later step by the method
    ``core.util.parse``).

    This function applies several simplifications and adaptations to prepare
    the model for conversion to spiking. These modifications include:

    - Removing layers only used during training (Dropout, BatchNormalization,
      ...)
    - Absorbing the parameters of BatchNormalization layers into the parameters
      of the preceeding layer. This does not affect performance because
      batch-norm-parameters are constant at inference time.
    - Removing ReLU activation layers, because their function is inherent to
      the spike generation mechanism. The information which nonlinearity was
      used in the original model is preserved in the layer specifications of
      the parsed model. If the output layer employs the softmax function, a
      spiking version is used when testing the SNN in INIsim or MegaSim
      simulators.
    - Inserting a Flatten layer between Conv and FC layers, if the input model
      did not explicitly include one.

    Parameters
    ----------

    model: lasagne.layers.Layer
        Lasagne model instance of the network, given by the last layer.
        Obtained from calling the ``load_ann`` function in this module.

    Returns
    -------

    Dictionary containing the parsed network specifications.

    layers: list
        List of all the layers of the network, where each layer contains a
        dictionary with keys

        - layer_type (string): Describing the type, e.g. `Dense`,
          `Convolution`, `Pool`.

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
    """

    lasagne_layers = lasagne.layers.get_all_layers(model)
    all_parameters = lasagne.layers.get_all_param_values(model)

    layers = []
    parameters_idx = 0
    idx = 0
    for (layer_num, layer) in enumerate(lasagne_layers):

        # Convert Lasagne layer names to our 'standard' names.
        name = layer.__class__.__name__
        if name == 'Pool2DLayer' and layer.mode == 'max':
            name = 'MaxPool2DLayer'
        layer_type = layer_dict.get(name, name)

        attributes = {'layer_type': layer_type}

        if layer_type == 'BatchNormalization':
            bn_parameters = all_parameters[parameters_idx: parameters_idx + 4]
            for k in range(1, 3):
                prev_layer = layers[-k]
                if 'parameters' in prev_layer:
                    break
            parameters = prev_layer['parameters']
            print("Absorbing batch-normalization parameters into " +
                  "parameters of previous {}.".format(prev_layer['name']))
            assert isinstance(layer, object)
            prev_layer['parameters'] = absorb_bn(
                parameters[0], parameters[1], bn_parameters[1],
                bn_parameters[0], bn_parameters[2], bn_parameters[3])
            parameters_idx += 4

        if layer_type not in spiking_layers:
            print("Skipping layer {}".format(layer_type))
            continue

        print("Parsing layer {}".format(layer_type))

        if idx == 0:
            batch_input_shape = list(layer.input_shape)
            batch_input_shape[0] = settings['batch_size']
            attributes['batch_input_shape'] = tuple(batch_input_shape)

        # Insert Flatten layer
        output_shape = lasagne_layers[layer_num].output_shape
        prev_layer_output_shape = lasagne_layers[layer_num-1].output_shape
        if len(output_shape) < len(prev_layer_output_shape) and \
                layer_type != 'Flatten':
            print("Inserting layer Flatten")
            # Append layer label
            num_str = str(idx) if idx > 9 else '0' + str(idx)
            shape_string = str(np.prod(output_shape[1:]))
            layers.append({'name': num_str + 'Flatten_' + shape_string,
                           'layer_type': 'Flatten'})
            idx += 1

        # Append layer label
        if len(output_shape) == 2:
            shape_string = '_{}'.format(output_shape[1])
        else:
            shape_string = '_{}x{}x{}'.format(
                output_shape[1], output_shape[2], output_shape[3])
        num_str = str(idx) if idx > 9 else '0' + str(idx)
        attributes['name'] = num_str + layer_type + shape_string

        if layer_type in {'Dense', 'Convolution2D'}:
            attributes['parameters'] = all_parameters[parameters_idx:
                                                      parameters_idx + 2]
            if settings['binarize_weights']:
                print("Binarizing weights...")
                attributes['parameters'] = \
                    (binarize(attributes['parameters'][0]),
                     attributes['parameters'][1])
            parameters_idx += 2  # For weights and biases
            # Get type of nonlinearity if the activation is directly in the
            # Dense / Conv layer:
            activation = activation_dict.get(layer.nonlinearity.__name__,
                                             'linear')
            # Otherwise, search for the activation layer:
            for k in range(layer_num+1, min(layer_num+4, len(lasagne_layers))):
                if lasagne_layers[k].__class__.__name__ == 'NonlinearityLayer':
                    nonlinearity = lasagne_layers[k].nonlinearity.__name__
                    activation = activation_dict.get(nonlinearity, 'linear')
                    break
            attributes['activation'] = activation
            print("Detected activation {}".format(activation))
            if layer_type == 'Convolution2D':
                border_mode = border_mode_string(layer.pad, layer.filter_size)
                attributes.update({'input_shape': layer.input_shape,
                                   'nb_filter': layer.num_filters,
                                   'nb_col': layer.filter_size[1],
                                   'nb_row': layer.filter_size[0],
                                   'border_mode': border_mode,
                                   'subsample': layer.stride,
                                   'filter_flip': layer.flip_filters})
            else:
                attributes['output_dim'] = layer.num_units

        if layer_type in {'MaxPooling2D', 'AveragePooling2D'}:
            border_mode = border_mode_string(layer.pad, layer.pool_size)
            attributes.update({'input_shape': layer.input_shape,
                               'pool_size': layer.pool_size,
                               'strides': layer.stride,
                               'border_mode': border_mode})
        # Append layer
        layers.append(attributes)
        idx += 1

    return layers


def hard_sigmoid(x):
    """Hard sigmoid (step) function.

    Parameters
    ----------
    x: np.array
        Input values.

    Returns
    -------

    : np.array
        Array with values in ``{0, 1}``
    """

    return np.clip(np.divide((x + 1.), 2.), 0, 1)


def binarize(w, h=1., deterministic=True):
    """Binarize weights.

    Parameters
    ----------
    w: np.array
        Weights.
    h: float
        Values are round to ``+/-h``.
    deterministic: bool
        Whether to apply deterministic rounding.

    Returns
    -------

    : np.array
        The binarized weights.
    """

    wb = hard_sigmoid(w / h)
    # noinspection PyTypeChecker
    wb = np.round(wb) if deterministic else np.random.binomial(1, wb)
    wb[wb.nonzero()] = h
    wb[wb == 0] = -h
    return np.asarray(wb, theano.config.floatX)


def absorb_bn(w, b, gamma, beta, mean, var_squ_eps_inv):
    """
    Absorb the parameters of a batch-normalization layer into the previous
    layer.
    """

    axis = 0 if w.ndim > 2 else 1

    broadcast_shape = [1] * w.ndim  # e.g. [1, 1, 1, 1] for ConvLayer
    broadcast_shape[axis] = w.shape[axis]  # [64, 1, 1, 1] for 64 features
    var_squ_eps_inv_broadcast = np.reshape(var_squ_eps_inv, broadcast_shape)
    gamma_broadcast = np.reshape(gamma, broadcast_shape)

    b_bn = beta + (b - mean) * gamma * var_squ_eps_inv
    w_bn = w * gamma_broadcast * var_squ_eps_inv_broadcast

    return w_bn, b_bn


def load_ann(path=None, filename=None):
    """Load network from file.

    Parameters
    ----------

    path: Optional[str]
        Path to directory where to load model from. Defaults to
        ``settings['path']``.
    filename: Optional[str]
        Name of file to load model from. Defaults to ``settings['filename']``.

    Returns
    -------

    : dict[str, Union[lasagne.models, theano.function]]
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

    return model_from_py(path, filename)


def model_from_py(path=None, filename=None):
    """Load model from *.py file.

    Parameters
    ----------

    path: Optional[str]
        Path to directory where to load model from. Defaults to
        ``settings['path']``.
    filename: Optional[str]
        Name of file to load model from. Defaults to ``settings['filename']``.

    Returns
    -------

    : dict[str, Union[lasagne.layers.Layer, theano.function]]
        A dictionary of objects that constitute the input model. It must
        contain the following two keys:

        - 'model': lasagne.layers.Layer
            Lasagne model instance of the network, given by the last layer.
        - 'val_fn': Theano function that allows evaluating the original
          model.
    """

    if path is None:
        path = settings['path']
    if filename is None:
        filename = settings['filename']

    mod = import_script(path, filename)
    model, train_fn, val_fn = mod.build_network()
    params = load_parameters(os.path.join(path, filename + '.h5'))
    lasagne.layers.set_all_param_values(model, params)

    return {'model': model, 'val_fn': val_fn}


def evaluate(val_fn, x_test=None, y_test=None, dataflow=None):
    """Evaluate the original ANN.

    Can use either numpy arrays ``x_test, y_test`` containing the test samples,
    or generate them with a dataflow
    (``Keras.ImageDataGenerator.flow_from_directory`` object).
    """

    err = 0
    loss = 0

    if x_test is None:
        # Get samples from Keras.ImageDataGenerator
        batch_size = dataflow.batch_size
        dataflow.batch_size = settings['num_to_test']
        x_test, y_test = dataflow.next()
        dataflow.batch_size = batch_size
        print("Using {} samples to evaluate input model".format(len(x_test)))

    batch_size = settings['batch_size']
    batches = int(len(x_test) / batch_size)

    for i in range(batches):
        new_loss, new_err = val_fn(x_test[i*batch_size: (i+1)*batch_size],
                                   y_test[i*batch_size: (i+1)*batch_size])
        err += new_err
        loss += new_loss

    err /= batches
    loss /= batches
    acc = 1 - err  # Convert error into accuracy here.

    print('\n' + "Test loss: {:.2f}".format(loss))
    print("Test accuracy: {:.2%}\n".format(acc))

    return loss, acc
