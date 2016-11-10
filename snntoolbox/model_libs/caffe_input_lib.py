# -*- coding: utf-8 -*-
"""
Methods to extract an input model written in caffe and prepare it for further
processing in the SNN toolbox.

The idea is to make all further steps in the conversion/simulation pipeline
independent of the original model format. Therefore, when a developer adds a
new input model library (e.g. Torch) to the toolbox, the following methods must
be implemented and satisfy the return requirements specified in their
respective docstrings:

    - extract
    - load_ann
    - evaluate

Created on Thu Jun  9 08:11:09 2016

@author: rbodo
"""

import os

import caffe
import numpy as np
from snntoolbox.config import settings, spiking_layers
from snntoolbox.model_libs.common import absorb_bn, border_mode_string

caffe.set_mode_gpu()

layer_dict = {'InnerProduct': 'Dense',
              'Convolution': 'Convolution2D',
              'MaxPooling2D': 'MaxPooling2D',
              'AveragePooling2D': 'AveragePooling2D',
              'ReLU': 'Activation',
              'Softmax': 'Activation'}


activation_dict = {'ReLU': 'relu',
                   'Softmax': 'softmax',
                   'Sigmoid': 'sigmoid'}


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

    model: dict
        A dictionary of objects that constitute the input model. Contains at
        least the key
            - ``model``: A model instance of the network in the respective
              ``model_lib``.
            - ``model_protobuf``: caffe.proto.caffe_pb2.NetParameter protocol
              buffer. The result of reading out the network specification from
              the prototxt file.
        For instance, if the input model was written using Keras, the 'model'-
        value would be an instance of ``keras.Model``.

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

    caffe_model = model[0]
    caffe_layers = model[1].layer

    batch_input_shape = list(model[1].input_dim)
    batch_input_shape[0] = settings['batch_size']

    layers = []
    idx = 0
    for (layer_num, layer) in enumerate(caffe_layers):

        # Convert Caffe layer names to our 'standard' names.
        name = layer.type
        if name == 'Pooling':
            pooling = layer.pooling_param.PoolMethod.DESCRIPTOR.values[0].name
            name = 'MaxPooling2D' if pooling == 'MAX' else 'AveragePooling2D'
        layer_type = layer_dict.get(name, name)

        attributes = {'layer_type': layer_type}

        if layer_type == 'BatchNormalization':
            bn_parameters = [layer.blobs[0].data,
                             layer.blobs[1].data]
            for k in range(1, 3):
                prev_layer = layers[-k]
                if 'parameters' in prev_layer:
                    break
            parameters = prev_layer['parameters']
            print("Absorbing batch-normalization parameters into " +
                  "parameters of previous {}.".format(prev_layer['name']))
            prev_layer['parameters'] = absorb_bn(
                parameters[0], parameters[1], bn_parameters[1],
                bn_parameters[0], bn_parameters[2], 1 / bn_parameters[3],
                layer.epsilon)

        if layer_type not in spiking_layers:
            print("Skipping layer {}".format(layer_type))
            continue

        print("Parsing layer {}".format(layer_type))

        if idx == 0:
            attributes['batch_input_shape'] = tuple(batch_input_shape)

        # Insert Flatten layer
        prev_layer_key = None
        output_shape = list(caffe_model.blobs[layer.name].shape)
        for k in [layer_num - i for i in range(1, 4)]:
            prev_layer_key = caffe_layers[k].name
            if prev_layer_key in caffe_model.blobs:
                break
        assert prev_layer_key, "Search for layer to flatten was unsuccessful."
        prev_layer_output_shape = list(caffe_model.blobs[prev_layer_key].shape)
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
            w = caffe_model.params[layer.name][0].data
            b = caffe_model.params[layer.name][1].data
            if layer_type == 'Dense':
                w = np.transpose(w)
            attributes['parameters'] = [w, b]
            # Get type of nonlinearity if the activation is directly in the
            # Dense / Conv layer:
            activation = activation_dict.get(layer.__class__.__name__,
                                             'linear')
            # Otherwise, search for the activation layer:
            for k in range(layer_num+1, min(layer_num+4, len(caffe_layers))):
                nonlinearity = caffe_layers[k].__class__.__name__
                if nonlinearity in {'ReLU', 'Softmax'}:
                    activation = activation_dict.get(nonlinearity, 'linear')
                    break
            attributes['activation'] = activation
            print("Detected activation {}".format(activation))

        if layer_type == 'Convolution2D':
            p = layer.convolution_param
            # Take maximum here because sometimes not not all fields are set
            # (e.g. kernel_h == 0 even though kernel_size == [3])
            filter_size = [max(p.kernel_w, p.kernel_size[0]),
                           max(p.kernel_h, p.kernel_size[-1])]
            pad = (p.pad_w, p.pad_h)
            border_mode = border_mode_string(pad, filter_size)
            attributes.update({'nb_filter': p.num_output,
                               'nb_col': filter_size[0],
                               'nb_row': filter_size[1],
                               'border_mode': border_mode,
                               'subsample': (p.stride[0], p.stride[0]),
                               'filter_flip': False})  # p.filter_flip

        if layer_type == 'Dense':
            attributes['output_dim'] = layer.inner_product_param.num_output

        if layer_type in {'MaxPooling2D', 'AveragePooling2D'}:
            p = layer.pooling_param
            # Take maximum here because sometimes not not all fields are set
            # (e.g. kernel_h == 0 even though kernel_size == 2)
            pool_size = [max(p.kernel_w, p.kernel_size),
                         max(p.kernel_h, p.kernel_size)]
            pad = (max(p.pad_w, p.pad), max(p.pad_h, p.pad))
            border_mode = border_mode_string(pad, pool_size)
            strides = [max(p.stride_w, p.stride), max(p.stride_h, p.stride)]
            attributes.update({'pool_size': pool_size,
                               'strides': strides,
                               'border_mode': border_mode})
        # Append layer
        layers.append(attributes)
        idx += 1

    return layers


def load_ann(path=None, filename=None):
    """Load network from file.

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

    return {'model': (model, model_protobuf), 'val_fn': model.forward_all}


def evaluate(val_fn, x_test=None, y_test=None, dataflow=None):
    """Evaluate the original ANN.

    Can use either numpy arrays ``x_test, y_test`` containing the test samples,
    or generate them with a dataflow
    (``Keras.ImageDataGenerator.flow_from_directory`` object).
    """

    if x_test is None:
        # Get samples from Keras.ImageDataGenerator
        batch_size = dataflow.batch_size
        dataflow.batch_size = settings['num_to_test']
        x_test, y_test = dataflow.next()
        dataflow.batch_size = batch_size
        print("Using {} samples to evaluate input model".format(len(x_test)))

    guesses = np.argmax(val_fn(data=x_test)['prob'], axis=1)
    truth = np.argmax(y_test, axis=1)
    accuracy = np.mean(guesses == truth)
    loss = -1

    print('\n' + "Test loss: {:.2f}".format(loss))
    print("Test accuracy: {:.2%}\n".format(accuracy))

    return [loss, accuracy]
