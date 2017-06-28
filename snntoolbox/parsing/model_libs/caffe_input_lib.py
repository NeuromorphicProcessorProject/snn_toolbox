# -*- coding: utf-8 -*-
"""Caffe model parser.

@author: rbodo
"""

import numpy as np

from snntoolbox.parsing.utils import AbstractModelParser, padding_string


class ModelParser(AbstractModelParser):

    def __init__(self, input_model, config):
        AbstractModelParser.__init__(self, input_model, config)
        self._layer_dict = {'InnerProduct': 'Dense',
                            'Convolution': 'Conv2D',
                            'MaxPooling2D': 'MaxPooling2D',
                            'AveragePooling2D': 'AveragePooling2D',
                            'ReLU': 'Activation',
                            'Softmax': 'Activation',
                            'Concat': 'Concatenate'}
        self.activation_dict = {'ReLU': 'relu',
                                'Softmax': 'softmax',
                                'Sigmoid': 'sigmoid'}

    def get_layer_iterable(self):
        return self.input_model[1].layer

    def get_type(self, layer):
        class_name = layer.type
        if class_name == 'Pooling':
            pooling = layer.pooling_param.PoolMethod.DESCRIPTOR.values[0].name
            class_name = 'MaxPooling2D' if pooling == 'MAX' \
                else 'AveragePooling2D'
        return self._layer_dict.get(class_name, class_name)

    def get_batchnorm_parameters(self, layer):
        raise NotImplementedError

    def get_inbound_layers(self, layer):
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

        layers = self.get_layer_iterable()
        inbound = []
        for inb in layer.bottom:  # Contains only labels
            for l in layers:
                name = 'data' if l.name == 'input' else l.name
                if inb == name:
                    inbound.append(l)
                    break
        return inbound

    def get_input_shape(self):
        return tuple(self.get_layer_iterable()[0].input_param.shape[0].dim[1:])

    def get_output_shape(self, layer):
        try:
            name = 'data' if layer.name == 'input' else layer.name
            return tuple(self.input_model[0].blobs[name].shape)
        except (KeyError, TypeError):
            print("Can't get output_shape because layer has no blobs.")

    def has_weights(self, layer):
        return len(self.input_model[0].params[layer.name])

    def parse_dense(self, layer, attributes):
        weights = np.transpose(self.input_model[0].params[layer.name][0].data)
        bias = self.input_model[0].params[layer.name][1].data
        if bias is None:
            bias = np.zeros(layer.num_output)
        attributes['parameters'] = [weights, bias]
        attributes['units'] = layer.inner_product_param.num_output

    def parse_convolution(self, layer, attributes):
        weights = self.input_model[0].params[layer.name][0].data
        bias = self.input_model[0].params[layer.name][1].data
        weights = weights[:, :, ::-1, ::-1]
        weights = np.transpose(weights, (2, 3, 1, 0))
        print("Flipped kernels.")
        attributes['parameters'] = [weights, bias]
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
                           'strides': (p.stride[0], p.stride[0])})

    def parse_pooling(self, layer, attributes):
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

    def get_activation(self, layer):
        return self.activation_dict.get(layer.type, 'linear')

    def get_outbound_layers(self, layer):
        layers = self.get_layer_iterable()
        layer_ids = [id(l) for l in layers]
        current_idx = layer_ids.index(id(layer))
        return [] if current_idx + 1 >= len(layer_ids) \
            else [layers[current_idx + 1]]

    def parse_concatenate(self, layer, attributes):
        attributes.update({'mode': 'concat',
                           'concat_axis': layer.concat_param.axis})


def load(path=None, filename=None):
    """Load network from file.

    Parameters
    ----------

    path: str
        Path to directory where to load model from.
    filename: str
        Name of file to load model from.

    Returns
    -------

    model: dict
        A dictionary of objects that constitute the input model. It must
        contain the following two keys:

        - 'model': tuple[caffe.Net, caffe.proto.caffe_pb2.NetParameter]
            Caffe model instance.
        - 'val_fn':
            Function that allows evaluating the original model.
    """

    import os
    from google.protobuf import text_format
    import caffe
    caffe.set_mode_gpu()

    prototxt = os.path.join(path, filename + '.prototxt')
    caffemodel = os.path.join(path, filename + '.caffemodel')
    model = caffe.Net(prototxt, 1, weights=caffemodel)
    model_protobuf = caffe.proto.caffe_pb2.NetParameter()
    text_format.Merge(open(prototxt).read(), model_protobuf)

    return {'model': (model, model_protobuf), 'val_fn': model.forward_all}


def evaluate(val_fn, batch_size, num_to_test, x_test=None, y_test=None,
             dataflow=None):
    """Evaluate the original ANN.

    Can use either numpy arrays ``x_test, y_test`` containing the test samples,
    or generate them with a dataflow
    (``Keras.ImageDataGenerator.flow_from_directory`` object).

    Parameters
    ----------

    val_fn:
        Function to evaluate model.

    batch_size: int
        Batch size

    num_to_test: int
        Number of samples to test

    x_test: Optional[np.ndarray]

    y_test: Optional[np.ndarray]

    dataflow: keras.ImageDataGenerator.flow_from_directory
    """

    accuracy = 0
    batches = int(len(x_test) / batch_size) if x_test is not None else \
        int(num_to_test / batch_size)

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

    return accuracy
