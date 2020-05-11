# -*- coding: utf-8 -*-
"""Keras model parser.

@author: rbodo
"""
import os

import numpy as np
import tensorflow.keras.backend as k
from tensorflow.keras import models, metrics

from snntoolbox.parsing.utils import AbstractModelParser, has_weights, \
    fix_input_layer_shape, get_type, get_inbound_layers, get_outbound_layers, \
    get_custom_activations_dict, assemble_custom_dict, get_custom_layers_dict


class ModelParser(AbstractModelParser):

    def get_layer_iterable(self):
        return self.input_model.layers

    def get_type(self, layer):
        return get_type(layer)

    def get_batchnorm_parameters(self, layer):
        mean = k.get_value(layer.moving_mean)
        var = k.get_value(layer.moving_variance)
        var_eps_sqrt_inv = 1 / np.sqrt(var + layer.epsilon)
        gamma = np.ones_like(mean) if layer.gamma is None else \
            k.get_value(layer.gamma)
        beta = np.zeros_like(mean) if layer.beta is None else \
            k.get_value(layer.beta)
        axis = layer.axis
        if isinstance(axis, (list, tuple)):
            assert len(axis) == 1, "Multiple BatchNorm axes not understood."
            axis = axis[0]

        return [mean, var_eps_sqrt_inv, gamma, beta, axis]

    def get_inbound_layers(self, layer):
        return get_inbound_layers(layer)

    @property
    def layers_to_skip(self):
        # noinspection PyArgumentList
        return AbstractModelParser.layers_to_skip.fget(self)

    def has_weights(self, layer):
        return has_weights(layer)

    def initialize_attributes(self, layer=None):
        attributes = AbstractModelParser.initialize_attributes(self)
        attributes.update(layer.get_config())
        return attributes

    def get_input_shape(self):
        return \
            fix_input_layer_shape(self.get_layer_iterable()[0].input_shape)[1:]

    def get_output_shape(self, layer):
        return layer.output_shape

    def parse_sparse(self, layer, attributes):
        return self.parse_dense(layer, attributes)

    def parse_dense(self, layer, attributes):
        attributes['parameters'] = list(layer.get_weights())
        if layer.bias is None:
            attributes['parameters'].insert(
                1, np.zeros(layer.output_shape[1]))
            attributes['parameters'] = tuple(attributes['parameters'])
            attributes['use_bias'] = True

    def parse_sparse_convolution(self, layer, attributes):
        return self.parse_convolution(layer, attributes)

    def parse_convolution(self, layer, attributes):
        attributes['parameters'] = list(layer.get_weights())
        if layer.bias is None:
            attributes['parameters'].insert(1, np.zeros(layer.filters))
            attributes['parameters'] = tuple(attributes['parameters'])
            attributes['use_bias'] = True
        assert layer.data_format == k.image_data_format(), (
            "The input model was setup with image data format '{}', but your "
            "keras config file expects '{}'.".format(layer.data_format,
                                                     k.image_data_format()))

    def parse_sparse_depthwiseconvolution(self, layer, attributes):
        return self.parse_depthwiseconvolution(layer, attributes)

    def parse_depthwiseconvolution(self, layer, attributes):
        attributes['parameters'] = list(layer.get_weights())
        if layer.bias is None:
            a = 1 if layer.data_format == 'channels_first' else -1
            attributes['parameters'].insert(1, np.zeros(
                layer.depth_multiplier * layer.input_shape[a]))
            attributes['parameters'] = tuple(attributes['parameters'])
            attributes['use_bias'] = True

    def parse_pooling(self, layer, attributes):
        pass

    def get_activation(self, layer):

        return layer.activation.__name__

    def get_outbound_layers(self, layer):

        return get_outbound_layers(layer)

    def parse_concatenate(self, layer, attributes):
        pass


def load(path, filename, **kwargs):
    """Load network from file.

    Parameters
    ----------

    path: str
        Path to directory where to load model from.

    filename: str
        Name of file to load model from.

    Returns
    -------

    : dict[str, Union[keras.models.Sequential, function]]
        A dictionary of objects that constitute the input model. It must
        contain the following two keys:

        - 'model': keras.models.Sequential
            Keras model instance of the network.
        - 'val_fn': function
            Function that allows evaluating the original model.
    """

    filepath = str(os.path.join(path, filename))

    if os.path.exists(filepath + '.json'):
        model = models.model_from_json(open(filepath + '.json').read())
        try:
            model.load_weights(filepath + '.h5')
        except OSError:
            # Allows h5 files without a .h5 extension to be loaded.
            model.load_weights(filepath)
        # With this loading method, optimizer and loss cannot be recovered.
        # Could be specified by user, but since they are not really needed
        # at inference time, set them to the most common choice.
        # TODO: Proper reinstantiation should be doable since Keras2
        model.compile('sgd', 'categorical_crossentropy',
                      ['accuracy', metrics.top_k_categorical_accuracy])
    else:
        filepath_custom_objects = kwargs.get('filepath_custom_objects', None)
        if filepath_custom_objects is not None:
            filepath_custom_objects = str(filepath_custom_objects)  # python 2

        custom_dicts = assemble_custom_dict(
            get_custom_activations_dict(filepath_custom_objects),
            get_custom_layers_dict())
        try:
            model = models.load_model(filepath + '.h5', custom_dicts)
        except OSError as e:
            print(e)
            print("Trying to load without '.h5' extension.")
            model = models.load_model(filepath, custom_dicts)
        model.compile(model.optimizer, model.loss,
                      ['accuracy', metrics.top_k_categorical_accuracy])

    model.summary()
    return {'model': model, 'val_fn': model.evaluate}


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

    if x_test is not None:
        score = val_fn(x_test, y_test, batch_size, verbose=0)
    else:
        score = np.zeros(3)
        batches = int(num_to_test / batch_size)
        for i in range(batches):
            x_batch, y_batch = dataflow.next()
            score += val_fn(x_batch, y_batch, batch_size, verbose=0)
        score /= batches

    print("Top-1 accuracy: {:.2%}".format(score[1]))
    print("Top-5 accuracy: {:.2%}\n".format(score[2]))

    return score[1]
