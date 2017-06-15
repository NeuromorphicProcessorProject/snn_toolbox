# -*- coding: utf-8 -*-
"""Functions common to input model parsers.

AbstractModelParser extracts an input model written in some neural
network library and prepares it for further processing in the SNN toolbox.

The idea is to make all further steps in the conversion/simulation pipeline
independent of the original model format.

Created on Thu May 19 08:26:49 2016

@author: rbodo
"""

import numpy as np
import keras
from abc import abstractmethod


class AbstractModelParser:
    """Abstract base class for neural network model parsers.

    Attributes
    ----------

    layer_list: list[dict]
        A list where each entry is a dictionary containing layer
        specifications. Can be obtained by calling
        ``model_lib.extract(input_model)`` (see ``snntoolbox.model_libs``
        package).

    """

    def __init__(self, input_model, config):
        self.input_model = input_model
        self.config = config
        self.layer_list = []
        self.layer_dict = {}
        self.parsed_model = None

    def parse(self):
        """Extract the essential information about a neural network.

        This method serves to abstract the conversion process of a network
        from the
        language the input model was built in (e.g. Keras or Lasagne).

        To extend the toolbox by another input format (e.g. Caffe),
        this method has
        to be implemented for the respective model library.

        Implementation details:
        The methods iterates over all layers of the input model and writes the
        layer specifications and parameters into a dictionary. The keys are
        chosen
        in accordance with Keras layer attributes to facilitate instantiation
        of a
        new, parsed Keras model (done in a later step by the method
        ``core.util.parse``).

        This function applies several simplifications and adaptations to prepare
        the model for conversion to spiking. These modifications include:

        - Removing layers only used during training (Dropout,
        BatchNormalization,
          ...)
        - Absorbing the parameters of BatchNormalization layers into the
        parameters
          of the preceeding layer. This does not affect performance because
          batch-norm-parameters are constant at inference time.
        - Removing ReLU activation layers, because their function is inherent to
          the spike generation mechanism. The information which nonlinearity was
          used in the original model is preserved in the layer specifications of
          the parsed model. If the output layer employs the softmax function, a
          spiking version is used when testing the SNN in INIsim or MegaSim
          simulators.
        - Inserting a Flatten layer between Conv and FC layers, if the input
        model
          did not explicitly include one.

        Returns
        -------

        Dictionary containing the parsed network specifications.

        layers: list
            List of all the layers of the network, where each layer contains a
            dictionary with keys

            - layer_type (string): Describing the type, e.g. `Dense`,
              `Convolution`, `Pool`.

            In addition, `Dense` and `Convolution` layer types contain

            - parameters (array): The weights and biases connecting this
            layer with
              the previous.

            `Convolution` layers contain further

            - kernel_size (tuple/list of 2 ints): The x- and y-dimension of
            filters.
            - padding (string): How to handle borders during convolution, e.g.
              `full`, `valid`, `same`.

            `Pooling` layers contain

            - pool_size (list): Specifies the subsampling factor in each
            dimension.
            - strides (list): The stepsize in each dimension during pooling.
        """

        from snntoolbox.config import snn_layers

        layers = self.get_layer_iterable()

        name_map = {}
        idx = 0
        inserted_flatten = False
        for layer in layers:
            layer_type = self.get_type(layer)

            # Absorb BatchNormalization layer into parameters of previous layer
            if layer_type == 'BatchNormalization':
                parameters_bn = list(self.get_batchnorm_parameters(layer))
                inbound = self.get_inbound_layers_with_parameters(layer)
                assert len(inbound) == 1, \
                    "Could not find unique layer with parameters " \
                    "preceeding BatchNorm layer."
                prev_layer = inbound[0]
                prev_layer_idx = name_map[str(id(prev_layer))]
                parameters = list(self.layer_list[prev_layer_idx]['parameters'])
                print("Absorbing batch-normalization parameters into " +
                      "parameters of previous {}.".format(self.get_type(
                          prev_layer)))
                self.layer_list[prev_layer_idx]['parameters'] = \
                    absorb_bn_parameters(*(parameters + parameters_bn))

            if layer_type == 'GlobalAveragePooling2D':
                print("Replacing GlobalAveragePooling by AveragePooling "
                      "plus Flatten.")
                pool_size = [layer.input_shape[-2], layer.input_shape[-1]]
                self.layer_list.append(
                    {'layer_type': 'AveragePooling2D',
                     'name': self.get_name(layer, idx, 'AveragePooling2D'),
                     'input_shape': layer.input_shape, 'pool_size': pool_size,
                     'inbound': self.get_inbound_names(layer, name_map)})
                name_map['AveragePooling2D' + str(idx)] = idx
                idx += 1
                num_str = str(idx) if idx > 9 else '0' + str(idx)
                shape_string = str(np.prod(layer.output_shape[1:]))
                self.layer_list.append(
                    {'name': num_str + 'Flatten_' + shape_string,
                     'layer_type': 'Flatten',
                     'inbound': [self.layer_list[-1]['name']]})
                name_map['Flatten' + str(idx)] = idx
                idx += 1
                inserted_flatten = True

            if layer_type not in snn_layers:
                print("Skipping layer {}.".format(layer_type))
                continue

            if not inserted_flatten:
                inserted_flatten = self.try_insert_flatten(layer, idx, name_map)
                idx += inserted_flatten

            print("Parsing layer {}.".format(layer_type))

            if layer_type == 'MaxPooling2D' and \
                    self.config.getboolean('conversion', 'max2avg_pool'):
                print("Replacing max by average pooling.")
                layer_type = 'AveragePooling2D'

            if inserted_flatten:
                inbound = [self.layer_list[-1]['name']]
                inserted_flatten = False
            else:
                inbound = self.get_inbound_names(layer, name_map)

            attributes = self.initialize_attributes(layer)

            attributes.update({'layer_type': layer_type,
                               'name': self.get_name(layer, idx),
                               'inbound': inbound})

            if layer_type == 'Dense':
                self.parse_dense(layer, attributes)

            if layer_type == 'Conv2D':
                self.parse_convolution(layer, attributes)

            if layer_type in {'Dense', 'Conv2D'}:
                if self.config.getboolean('cell', 'binarize_weights'):
                    print("Binarizing weights.")
                    attributes['parameters'] = \
                        (binarize(attributes['parameters'][0]),
                         attributes['parameters'][1])

                self.absorb_activation(layer, attributes)

            if 'Pooling' in layer_type:
                self.parse_pooling(layer, attributes)

            if layer_type == 'Concatenate':
                self.parse_concatenate(layer, attributes)

            self.layer_list.append(attributes)

            # Map layer index to layer id. Needed for inception modules.
            name_map[str(id(layer))] = idx

            idx += 1
        print('')

    @abstractmethod
    def get_layer_iterable(self):
        """

        Returns
        -------

        layers: List
        """

        pass

    @abstractmethod
    def get_type(self, layer):
        """Get layer class name.

        Returns
        -------

        layer_type: str
            Layer class name.
        """

        pass

    @abstractmethod
    def get_batchnorm_parameters(self, layer):
        """

        Returns
        -------

        (mean, var_eps_sqrt_inv, gamma, beta, axis): iterable

        """

        pass

    def get_inbound_layers_with_parameters(self, layer):
        """Iterate until inbound layers are found that have parameters.

        Parameters
        ----------

        layer:
            Layer

        Returns
        -------

        : list
            List of inbound layers.
        """

        inbound = layer
        while True:
            inbound = self.get_inbound_layers(inbound)
            if len(inbound) == 1:
                inbound = inbound[0]
                if self.has_weights(inbound):
                    return [inbound]
            else:
                result = []
                for inb in inbound:
                    if self.has_weights(inb):
                        result.append(inb)
                    else:
                        result += self.get_inbound_layers_with_parameters(inb)
                return result

    def get_inbound_names(self, layer, name_map):
        """Get names of inbound layers.

        """

        if len(self.layer_list) == 0:
            return ['input']

        inbound = self.get_inbound_layers(layer)
        for ib in range(len(inbound)):
            for _ in range(len(self.layers_to_skip)):
                if self.get_type(inbound[ib]) in self.layers_to_skip:
                    inbound[ib] = self.get_inbound_layers(inbound[ib])[0]
                else:
                    break
        inb_idxs = [name_map[str(id(inb))] for inb in inbound]
        return [self.layer_list[i]['name'] for i in inb_idxs]

    @abstractmethod
    def get_inbound_layers(self, layer):
        """

        Returns
        -------

        inbound: Sequence
        """

        pass

    @property
    def layers_to_skip(self):
        """

        Returns
        -------

        self._layers_to_skip: List[str]
        """

        return ['BatchNormalization', 'Activation', 'Dropout']

    @abstractmethod
    def has_weights(self, layer):
        pass

    def initialize_attributes(self, layer=None):
        return {}

    @abstractmethod
    def get_input_shape(self):
        pass

    def get_batch_input_shape(self):
        input_shape = tuple(self.get_input_shape())
        batch_size = self.config.getint('simulation', 'batch_size')
        return (batch_size,) + input_shape

    def get_name(self, layer, idx, layer_type=None):
        if layer_type is None:
            layer_type = self.get_type(layer)

        output_shape = self.get_output_shape(layer)
        if len(output_shape) == 2:
            shape_string = '_{}'.format(output_shape[1])
        else:
            shape_string = '_{}x{}x{}'.format(output_shape[1],
                                              output_shape[2],
                                              output_shape[3])

        num_str = str(idx) if idx > 9 else '0' + str(idx)

        return num_str + layer_type + shape_string

    @abstractmethod
    def get_output_shape(self, layer):
        """Get output shape of layer.

        Returns
        -------

        output_shape: Sized
            Output shape of layer.
        """

        pass

    def try_insert_flatten(self, layer, idx, name_map):
        output_shape = self.get_output_shape(layer)
        previous_layers = self.get_inbound_layers(layer)
        prev_layer_output_shape = self.get_output_shape(previous_layers[0])
        if len(output_shape) < len(prev_layer_output_shape) and \
                self.get_type(layer) != 'Flatten':
            assert len(previous_layers) == 1, "Layer to flatten must be unique."
            print("Inserting layer Flatten.")
            num_str = str(idx) if idx > 9 else '0' + str(idx)
            shape_string = str(np.prod(prev_layer_output_shape[1:]))
            self.layer_list.append({
                'name': num_str + 'Flatten_' + shape_string,
                'layer_type': 'Flatten',
                'inbound': self.get_inbound_names(layer, name_map)})
            name_map['Flatten' + str(idx)] = idx
            return True
        else:
            return False

    @abstractmethod
    def parse_dense(self, layer, attributes):
        pass

    @abstractmethod
    def parse_convolution(self, layer, attributes):
        pass

    @abstractmethod
    def parse_pooling(self, layer, attributes):
        pass

    def absorb_activation(self, layer, attributes):
        activation = self.get_activation(layer)

        # Sometimes the Conv/Dense layer specifies its activation directly,
        # sometimes it is followed by a dedicated activation layer (possibly
        # with BatchNormalization in between). Here we try to find such a
        # activation layer.
        outbound = layer
        for _ in range(3):
            outbound = self.get_outbound_layers(outbound)
            if len(outbound) != 1:
                break
            else:
                outbound = outbound[0]
                if self.get_type(outbound) == 'Activation':
                    activation = self.get_activation(outbound)
                    break

        # Maybe change activation to custom type.
        if activation == 'binary_sigmoid':
            activation = binary_sigmoid
        elif activation == 'binary_tanh':
            activation = binary_tanh
        elif activation == 'softmax' and \
                self.config.getboolean('conversion', 'softmax_to_relu'):
            activation = 'relu'
            print("Replaced softmax by relu activation function.")

        print("Using activation {}.".format(activation))
        attributes['activation'] = activation

    @abstractmethod
    def get_activation(self, layer):
        pass

    @abstractmethod
    def get_outbound_layers(self, layer):
        """

        Returns
        -------

        outbound: List

        """

        pass

    @abstractmethod
    def parse_concatenate(self, layer, attributes):
        pass

    def build_parsed_model(self):
        """Create a Keras model suitable for conversion to SNN.

        This method uses a list of layer specifications to build a Keras model
        from it. The resulting model contains all essential information about
        the original network, independently of the model library in which the
        original network was built (e.g. Caffe). This makes the SNN toolbox
        stable against changes in input formats. Another advantage is
        extensibility: In order to add a new input language to the toolbox (e.g.
        Lasagne), a developer only needs to add a single module to
        ``model_libs`` package, implementing a number of methods (see the
        respective functions in 'keras_input_lib.py' for more details.)

        Returns
        -------

        parsed_model: keras.models.Model
            A Keras model, functionally equivalent to ``input_model``.
        """

        img_input = keras.layers.Input(batch_shape=self.get_batch_input_shape(),
                                       name='input')
        parsed_layers = {'input': img_input}
        print("Building parsed model...\n")
        for layer in self.layer_list:
            # Replace 'parameters' key with Keras key 'weights'
            if 'parameters' in layer:
                layer['weights'] = layer.pop('parameters')

            # Add layer
            parsed_layer = getattr(keras.layers, layer.pop('layer_type'))

            inbound = [parsed_layers[inb] for inb in layer.pop('inbound')]
            if len(inbound) == 1:
                inbound = inbound[0]
            parsed_layers[layer['name']] = parsed_layer(**layer)(inbound)

        print("Compiling parsed model...\n")
        self.parsed_model = keras.models.Model(img_input, parsed_layers[
            self.layer_list[-1]['name']])
        # Optimizer and loss do not matter because we only do inference.
        self.parsed_model.compile(
            'sgd', 'categorical_crossentropy',
            ['accuracy', keras.metrics.top_k_categorical_accuracy])

        return self.parsed_model

    def evaluate_parsed(self, batch_size, num_to_test, x_test=None,
                        y_test=None, dataflow=None):
        """Evaluate parsed Keras model.

        Can use either numpy arrays ``x_test, y_test`` containing the test
        samples, or generate them with a dataflow
        (``Keras.ImageDataGenerator.flow_from_directory`` object).

        Parameters
        ----------

        batch_size: int
            Batch size

        num_to_test: int
            Number of samples to test

        x_test: Optional[np.ndarray]

        y_test: Optional[np.ndarray]

        dataflow: keras.ImageDataGenerator.flow_from_directory
        """

        assert (x_test is not None and y_test is not None or dataflow is not
                None), "No testsamples provided."

        if x_test is not None:
            score = self.parsed_model.evaluate(x_test, y_test, batch_size,
                                               verbose=0)
        else:
            steps = int(num_to_test / batch_size)
            score = self.parsed_model.evaluate_generator(dataflow, steps)
        print("Top-1 accuracy: {:.2%}".format(score[1]))
        print("Top-5 accuracy: {:.2%}\n".format(score[2]))

        return score


def absorb_bn_parameters(weight, bias, mean, var_eps_sqrt_inv, gamma, beta,
                         axis):
    """
    Absorb the parameters of a batch-normalization layer into the previous
    layer.
    """

    # TODO: Due to some issue when porting a Keras1 GoogLeNet model to Keras2,
    # the axis is 1 when it should be -1. Need to find a way to avoid this hack.
    # if axis != -1:
    #     print("Warning: Specifying a batch-normalization axis other than the "
    #           "default (-1) has not been thoroughly tested yet.")
    axis = -1

    ndim = weight.ndim
    reduction_axes = list(range(ndim))
    del reduction_axes[axis]

    if sorted(reduction_axes) != list(range(ndim))[:-1]:
        broadcast_shape = [1] * ndim
        broadcast_shape[axis] = weight.shape[axis]
        var_eps_sqrt_inv = np.reshape(var_eps_sqrt_inv, broadcast_shape)
        gamma = np.reshape(gamma, broadcast_shape)

    bias_bn = beta + (bias - mean) * gamma * var_eps_sqrt_inv
    weight_bn = weight * gamma * var_eps_sqrt_inv

    return weight_bn, bias_bn


def padding_string(pad, pool_size):
    """Get string defining the border mode.

    Parameters
    ----------
    pad: tuple[int]
        Zero-padding in x- and y-direction.
    pool_size: list[int]
        Size of kernel.

    Returns
    -------

    padding: str
        Border mode identifier.
    """

    if pad == (0, 0):
        padding = 'valid'
    elif pad == (pool_size[0] // 2, pool_size[1] // 2):
        padding = 'same'
    elif pad == (pool_size[0] - 1, pool_size[1] - 1):
        padding = 'full'
    else:
        raise NotImplementedError(
            "Padding {} could not be interpreted as any of the ".format(pad) +
            "supported border modes 'valid', 'same' or 'full'.")
    return padding


def import_script(path, filename):
    """Import python script independently from python version.

    Parameters
    ----------

    path: string
        Path to directory where to load script from.

    filename: string
        Name of script file.
    """

    import os
    import sys

    filepath = os.path.join(path, filename + '.py')

    v = sys.version_info
    if v >= (3, 5):
        import importlib.util
        spec = importlib.util.spec_from_file_location(filename, filepath)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    elif v >= (3, 3):
        # noinspection PyCompatibility,PyUnresolvedReferences
        from importlib.machinery import SourceFileLoader
        mod = SourceFileLoader(filename, filepath).load_module()
    else:
        # noinspection PyDeprecation
        import imp
        # noinspection PyDeprecation
        mod = imp.load_source(filename, filepath)
    return mod


def binary_tanh(x):
    """Round a float to -1 or 1.

    Parameters
    ----------

    x: float

    Returns
    -------

    : int
        Integer in {-1, 1}
    """

    return keras.backend.sign(x)


def binary_sigmoid(x):
    """Round a float to 0 or 1.

    Parameters
    ----------

    x: float

    Returns
    -------

    : int
        Integer in {0, 1}
    """

    return keras.backend.round(hard_sigmoid(x))


def hard_sigmoid(x):
    """

    Parameters
    ----------

    x :

    Returns
    -------

    """

    return keras.backend.clip((x + 1.) / 2., 0, 1)


def binarize_var(w, h=1., deterministic=True):
    """Binarize shared variable.

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

    # [-1, 1] -> [0, 1]
    wb = hard_sigmoid(w / h)

    # Deterministic / stochastic rounding
    wb = keras.backend.round(wb) if deterministic \
        else keras.backend.cast_to_floatx(np.random.binomial(1, wb))

    # {0, 1} -> {-1, 1}
    wb = keras.backend.cast_to_floatx(keras.backend.switch(wb, h, -h))

    return keras.backend.cast_to_floatx(wb)


def binarize(w, h=1., deterministic=True):
    """Binarize weights.

    Parameters
    ----------
    w: ndarray
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

    # [-1, 1] -> [0, 1]
    wb = np.clip((w / h + 1.) / 2., 0, 1)

    # Deterministic / stochastic rounding
    wb = np.round(wb) if deterministic else np.random.binomial(1, wb)

    # {0, 1} -> {-1, 1}
    wb[wb != 0] = h
    wb[wb == 0] = -h

    return np.asarray(wb, np.float32)
