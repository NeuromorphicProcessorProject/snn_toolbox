# -*- coding: utf-8 -*-
"""Lasagne model parser.

@author: rbodo
"""

import lasagne
import numpy as np

from snntoolbox.parsing.utils import AbstractModelParser, padding_string


class ModelParser(AbstractModelParser):

    def __init__(self, input_model, config):
        AbstractModelParser.__init__(self, input_model, config)
        self._layer_dict = {'DenseLayer': 'Dense',
                            'Conv2DLayer': 'Conv2D',
                            'Conv2DDNNLayer': 'Conv2D',
                            'MaxPool2DLayer': 'MaxPooling2D',
                            'Pool2DLayer': 'AveragePooling2D',
                            'DropoutLayer': 'Dropout',
                            'FlattenLayer': 'Flatten',
                            'BatchNormLayer': 'BatchNormalization',
                            'NonlinearityLayer': 'Activation',
                            'ConcatLayer': 'Concatenate',
                            'GlobalPoolLayer': 'GlobalAveragePooling2D'}
        self.activation_dict = {'rectify': 'relu',
                                'softmax': 'softmax',
                                'binary_tanh_unit': 'binary_tanh',
                                'binary_sigmoid_unit': 'binary_sigmoid',
                                'linear': 'linear'}

    def get_layer_iterable(self):
        return lasagne.layers.get_all_layers(self.input_model)

    def get_type(self, layer):
        class_name = layer.__class__.__name__
        if class_name == 'Pool2DLayer' and layer.mode == 'max':
            class_name = 'MaxPool2DLayer'
        return self._layer_dict.get(class_name, class_name)

    def get_batchnorm_parameters(self, layer):
        mean = layer.mean.get_value()
        var_eps_sqrt_inv = layer.inv_std.get_value()
        gamma = layer.gamma.get_value()
        beta = layer.beta.get_value()
        axis = None  # Lasagne: axes = (0, 2, 3). Keras: axis = 1. Transform:
        for axis, a in enumerate(layer.axes):
            if axis != a:
                break
        return [mean, var_eps_sqrt_inv, gamma, beta, axis]

    def get_inbound_layers(self, layer):
        """Return inbound layers.

        Parameters
        ----------

        layer: Union[lasagne.layers.Layer, lasagne.layers.MergeLayer]
            A Lasagne layer.

        Returns
        -------

        : list[lasagne.layers.Layer]
            List of inbound layers.
        """

        if hasattr(layer, 'input_layers'):
            return layer.input_layers
        return [layer.input_layer]

    def get_input_shape(self):
        return tuple(self.get_layer_iterable()[0].shape[1:])

    def get_output_shape(self, layer):
        return layer.output_shape

    def has_weights(self, layer):
        return len(layer.params)

    def parse_dense(self, layer, attributes):
        weights = layer.W.get_value()
        bias = layer.b.get_value()
        if bias is None:
            bias = np.zeros(layer.num_units)
        attributes['parameters'] = [weights, bias]
        attributes['units'] = layer.num_units

    def parse_convolution(self, layer, attributes):
        weights = layer.W.get_value()
        weights = np.transpose(weights, (2, 3, 1, 0))
        bias = layer.b.get_value()
        if bias is None:
            bias = np.zeros(layer.num_filters)
        attributes['parameters'] = [weights, bias]
        padding = padding_string(layer.pad, layer.filter_size)
        attributes.update({'input_shape': layer.input_shape,
                           'filters': layer.num_filters,
                           'kernel_size': layer.filter_size,
                           'padding': padding,
                           'strides': layer.stride})

    def parse_pooling(self, layer, attributes):
        attributes.update({
            'input_shape': layer.input_shape,
            'pool_size': layer.pool_size,
            'strides': layer.stride,
            'padding': padding_string(layer.pad, layer.pool_size)})

    def get_activation(self, layer):
        return self.activation_dict.get(layer.nonlinearity.__name__, 'linear')

    def get_outbound_layers(self, layer):
        layers = self.get_layer_iterable()
        layer_ids = [id(l) for l in layers]
        current_idx = layer_ids.index(id(layer))
        return [] if current_idx + 1 >= len(layer_ids) \
            else [layers[current_idx + 1]]

    def parse_concatenate(self, layer, attributes):
        attributes['axis'] = layer.axis


def load(path, filename):
    """Load network from file.

    Lasagne does not provide loading functions to restore a model from a saved
    file, so we use the script with which the model was compiled in the first
    place.

    Parameters
    ----------

    path: str
        Path to directory where to load model from.
    filename: str
        Name of file to load model from.

    Returns
    -------

    : dict[str, Union[lasagne.layers.Layer, function]]
        A dictionary of objects that constitute the input model. It must
        contain the following two keys:

        - 'model': lasagne.layers.Layer
            Lasagne model instance of the network, given by the last layer.
        - 'val_fn':
            Function that allows evaluating the original model.
    """

    import os
    from snntoolbox.utils.utils import import_script

    filepath = os.path.join(path, filename)

    mod = import_script(path, filename)
    model = mod.build_network()
    if os.path.isfile(filepath + '.pkl'):
        print("Loading parameters from .pkl file.")
        import pickle
        params = pickle.load(open(filepath + '.pkl', 'rb'),
                             encoding='latin1')['param values']
    else:
        print("Loading parameters from .h5 file.")
        from snntoolbox.parsing.utils import load_parameters
        params = load_parameters(filepath + '.h5')
    lasagne.layers.set_all_param_values(model['model'], params)

    return {'model': model['model'], 'val_fn': model['val_fn']}


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

    err = 0
    loss = 0
    batches = int(len(x_test) / batch_size) if x_test is not None else \
        int(num_to_test / batch_size)

    for i in range(batches):
        if x_test is not None:
            x_batch = x_test[i*batch_size: (i+1)*batch_size]
            y_batch = y_test[i*batch_size: (i+1)*batch_size]
        else:
            x_batch, y_batch = dataflow.next()
        new_loss, new_err = val_fn(x_batch, y_batch)
        err += new_err
        loss += new_loss

    err /= batches
    loss /= batches
    acc = 1 - err  # Convert error into accuracy here.

    print("Test loss: {:.2f}".format(loss))
    print("Test accuracy: {:.2%}\n".format(acc))

    return acc
