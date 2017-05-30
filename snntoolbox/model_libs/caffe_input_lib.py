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
from snntoolbox.model_libs.common import absorb_bn, padding_string

caffe.set_mode_gpu()

layer_dict = {'InnerProduct': 'Dense',
              'Convolution': 'Conv2D',
              'MaxPooling2D': 'MaxPooling2D',
              'AveragePooling2D': 'AveragePooling2D',
              'ReLU': 'Activation',
              'Softmax': 'Activation',
              'Concat': 'Concatenate'}


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

        - kernel_size (tuple/list of 2 ints): The x- and y-dimension of filters.
        - padding (string): How to handle borders during convolution, e.g.
          `full`, `valid`, `same`.

        `Pooling` layers contain

        - pool_size (list): Specifies the subsampling factor in each dimension.
        - strides (list): The stepsize in each dimension during pooling.
    """

    caffe_model = model[0]
    caffe_layers = model[1].layer

    batch_input_shape = list(caffe_layers[0].input_param.shape[0].dim)
    batch_input_shape[0] = None  # For flexibility; else: settings['batch_size']

    name_map = {}
    layers = []
    idx = 0
    inserted_flatten = False
    for (layer_num, layer) in enumerate(caffe_layers):

        # Convert Caffe layer names to our 'standard' names.
        name = layer.type
        if name == 'Pooling':
            pooling = layer.pooling_param.PoolMethod.DESCRIPTOR.values[0].name
            name = 'MaxPooling2D' if pooling == 'MAX' and not \
                settings['max2avg_pool'] else 'AveragePooling2D'
            print("Replacing Max by AveragePooling.")
        layer_type = layer_dict.get(name, name)

        attributes = {'layer_type': layer_type}

        if layer_type == 'BatchNormalization':
            bn_parameters = [layer.blobs[0].data, layer.blobs[1].data]
            inb = get_inbound_layers_with_params(layer)
            assert len(inb) == 1, "Could not find unique layer with " \
                                  "parameters preceeding BatchNorm layer."
            prev_layer = inb[0]
            prev_layer_idx = name_map[str(id(prev_layer))]
            parameters = layers[prev_layer_idx]['parameters']
            if len(parameters) == 1:  # No bias
                parameters.append(np.zeros_like(bn_parameters[0]))
            print("Absorbing batch-normalization parameters into " +
                  "parameters of previous {}.".format(prev_layer.name))
            layers[prev_layer_idx]['parameters'] = absorb_bn(
                parameters[0], parameters[1], bn_parameters[1],
                bn_parameters[0], bn_parameters[2], 1 / bn_parameters[3],
                layer.epsilon)

        if layer_type not in spiking_layers:
            print("Skipping layer {}.".format(layer_type))
            continue

        print("Parsing layer {}.".format(layer_type))

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
            print("Inserting layer Flatten.")
            num_str = str(idx) if idx > 9 else '0' + str(idx)
            shape_string = str(np.prod(output_shape[1:]))
            layers.append({'name': num_str + 'Flatten_' + shape_string,
                           'layer_type': 'Flatten', 'inbound':
                               get_inbound_names(layers, layer, name_map,
                                                 caffe_layers)})
            name_map[str(id(layer))] = idx
            idx += 1
            inserted_flatten = True

        # Append layer label
        if len(output_shape) == 2:
            shape_string = '_{}'.format(output_shape[1])
        else:
            shape_string = '_{}x{}x{}'.format(
                output_shape[1], output_shape[2], output_shape[3])
        num_str = str(idx) if idx > 9 else '0' + str(idx)
        attributes['name'] = num_str + layer_type + shape_string

        if layer_type in {'Dense', 'Conv2D'}:
            # Search for the activation layer to integrate it into Dense / Conv:
            activation = 'linear'
            for k in range(layer_num+1, min(layer_num+4, len(caffe_layers))):
                if caffe_layers[k].type in {'ReLU', 'Softmax'}:
                    activation = activation_dict.get(caffe_layers[k].type,
                                                     'linear')
                    break
            attributes['activation'] = activation
            print("Using activation {}.".format(activation))

        if layer_type == 'Conv2D':
            w = caffe_model.params[layer.name][0].data
            b = caffe_model.params[layer.name][1].data
            w = w[:, :, ::-1, ::-1]
            print("Flipped kernels.")
            w = np.transpose(w, (2, 3, 1, 0))
            attributes['parameters'] = [w, b]
            p = layer.convolution_param
            # Take maximum here because sometimes not not all fields are set
            # (e.g. kernel_h == 0 even though kernel_size == [3])
            filter_size = [max(p.kernel_w, p.kernel_size[0]),
                           max(p.kernel_h, p.kernel_size[-1])]
            pad = (p.pad_w, p.pad_h)
            padding = padding_string(pad, filter_size)
            attributes.update({'filters': p.num_output,
                               'kernel_size': filter_size,
                               'padding': padding,
                               'strides': (p.stride[0], p.stride[0]),
                               'filter_flip': False})  # p.filter_flip

        if layer_type == 'Dense':
            w = np.transpose(caffe_model.params[layer.name][0].data)
            b = caffe_model.params[layer.name][1].data
            attributes['parameters'] = [w, b]
            attributes['units'] = layer.inner_product_param.num_output

        if layer_type in {'MaxPooling2D', 'AveragePooling2D'}:
            p = layer.pooling_param
            # Take maximum here because sometimes not not all fields are set
            # (e.g. kernel_h == 0 even though kernel_size == 2)
            pool_size = [max(p.kernel_w, p.kernel_size),
                         max(p.kernel_h, p.kernel_size)]
            pad = (max(p.pad_w, p.pad), max(p.pad_h, p.pad))
            padding = padding_string(pad, pool_size)
            strides = [max(p.stride_w, p.stride), max(p.stride_h, p.stride)]
            attributes.update({'pool_size': pool_size,
                               'strides': strides,
                               'padding': padding})

        if layer_type == 'Concatenate':
            attributes.update({'mode': 'concat',
                               'concat_axis': layer.concat_param.axis})

        if inserted_flatten:
            attributes['inbound'] = [layers[-1]['name']]
            inserted_flatten = False
        else:
            attributes['inbound'] = get_inbound_names(layers, layer, name_map,
                                                      caffe_layers)

        # Append layer
        layers.append(attributes)
        # Map layer index to layer id. Needed for inception modules.
        name_map[str(id(layer))] = idx
        idx += 1
    print()

    return layers


def get_inbound_names(layers, layer, name_map, caffe_layers):
    """Get names of inbound layers.

    """

    if len(layers) == 0:
        return ['input_1']
    else:
        inbound_labels = get_inbound_layers(layer)
        inbound = []
        for il in inbound_labels:
            for l in caffe_layers:
                if il == l.name:
                    inbound.append(l)
        for ib in range(len(inbound)):
            ii = 0
            while ii < 3 and inbound[ib].type in \
                    ['BatchNorm', 'Dropout', 'ReLU', 'SoftmaxWithLoss']:
                inbound[ib] = get_inbound_layers(inbound[ib])[0]
                ii += 1
        inb_idxs = [name_map[str(id(inb))] for inb in inbound]
        return [layers[ii]['name'] for ii in inb_idxs]


def get_inbound_layers(layer):
    """Return inbound layers.

    Parameters
    ----------

    layer: Union[caffe.layers.Layer, caffe.layers.Concat]
        A Caffe layer.

    Returns
    -------

    : list[caffe.layers.Layer]
        List of inbound layers.
    """

    return layer.bottom


def get_inbound_layers_with_params(layer):
    """Iterate until inbound layers are found that have parameters.

    Parameters
    ----------

    layer: caffe.layers.Layer
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
            if len(inbound.blobs) > 0:
                return [inbound]
        else:
            result = []
            for inb in inbound:
                if len(inb.blobs) > 0:
                    result.append(inb)
                else:
                    result += get_inbound_layers_with_params(inb)
            return result


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
    model = caffe.Net(prototxt, 1, weights=caffemodel)
    model_protobuf = caffe.proto.caffe_pb2.NetParameter()
    text_format.Merge(open(prototxt).read(), model_protobuf)

    return {'model': (model, model_protobuf), 'val_fn': model.forward_all}


def evaluate(val_fn, x_test=None, y_test=None, dataflow=None):
    """Evaluate the original ANN.

    Can use either numpy arrays ``x_test, y_test`` containing the test samples,
    or generate them with a dataflow
    (``Keras.ImageDataGenerator.flow_from_directory`` object).
    """

    num_to_test = len(x_test) if x_test is not None else settings['num_to_test']
    print("Using {} samples to evaluate input model.".format(num_to_test))

    accuracy = 0
    batch_size = settings['batch_size']
    batches = int(len(x_test) / batch_size) if x_test is not None else \
        int(settings['num_to_test'] / batch_size)

    for i in range(batches):
        if x_test is not None:
            x_batch = x_test[i*batch_size: (i+1)*batch_size]
            y_batch = y_test[i*batch_size: (i+1)*batch_size]
        else:
            x_batch, y_batch = dataflow.next()
        out = val_fn(data=x_batch)
        guesses = np.argmax([out[key] for key in out.keys()][0], axis=1)
        truth = np.argmax(y_batch, axis=1)
        accuracy += np.mean(guesses == truth)

    accuracy /= batches

    print("Test accuracy: {:.2%}\n".format(accuracy))
