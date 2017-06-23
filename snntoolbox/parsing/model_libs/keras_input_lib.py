# -*- coding: utf-8 -*-
"""Keras model parser.

@author: rbodo
"""

import numpy as np

from snntoolbox.parsing.utils import AbstractModelParser


class ModelParser(AbstractModelParser):

    def get_layer_iterable(self):
        return self.input_model.layers

    def get_type(self, layer):
        from snntoolbox.parsing.utils import get_type
        return get_type(layer)

    def get_batchnorm_parameters(self, layer):
        mean = layer.moving_mean.get_value()
        var = layer.moving_variance.get_value()
        var_eps_sqrt_inv = 1 / np.sqrt(var + layer.epsilon)
        gamma = np.ones_like(mean) if layer.gamma is None else \
            layer.gamma.get_value()
        beta = np.zeros_like(mean) if layer.beta is None else \
            layer.beta.get_value()
        axis = layer.axis

        return [mean, var_eps_sqrt_inv, gamma, beta, axis]

    def get_inbound_layers(self, layer):
        from snntoolbox.parsing.utils import get_inbound_layers
        return get_inbound_layers(layer)

    @property
    def layers_to_skip(self):
        # noinspection PyArgumentList
        return AbstractModelParser.layers_to_skip.fget(self)

    def has_weights(self, layer):
        from snntoolbox.parsing.utils import has_weights
        return has_weights(layer)

    def initialize_attributes(self, layer=None):
        attributes = AbstractModelParser.initialize_attributes(self)
        attributes.update(layer.get_config())
        return attributes

    def get_input_shape(self):
        return tuple(self.get_layer_iterable()[0].batch_input_shape[1:])

    def get_output_shape(self, layer):
        return layer.output_shape

    def parse_dense(self, layer, attributes):
        attributes['parameters'] = layer.get_weights()
        if layer.bias is None:
            attributes['parameters'].append(np.zeros(layer.output_shape[1]))
            attributes['use_bias'] = True

    def parse_convolution(self, layer, attributes):
        attributes['parameters'] = layer.get_weights()
        if layer.bias is None:
            attributes['parameters'].append(np.zeros(layer.filters))
            attributes['use_bias'] = True

    def parse_pooling(self, layer, attributes):
        pass

    def get_activation(self, layer):

        return layer.activation.__name__

    def get_outbound_layers(self, layer):

        from snntoolbox.parsing.utils import get_outbound_layers

        return get_outbound_layers(layer)

    def parse_concatenate(self, layer, attributes):
        pass


def load(path, filename):
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

    import os
    from keras import models, metrics

    filepath = os.path.join(path, filename)

    if os.path.exists(filepath + '.json'):
        model = models.model_from_json(open(filepath + '.json').read())
        model.load_weights(filepath + '.h5')
        # With this loading method, optimizer and loss cannot be recovered.
        # Could be specified by user, but since they are not really needed
        # at inference time, set them to the most common choice.
        # TODO: Proper reinstantiation should be doable since Keras2
        model.compile('sgd', 'categorical_crossentropy',
                      ['accuracy', metrics.top_k_categorical_accuracy])
    else:
        model = models.load_model(filepath + '.h5')
        model.compile(model.optimizer, model.loss,
                      ['accuracy', metrics.top_k_categorical_accuracy])

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
