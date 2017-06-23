# -*- coding: utf-8 -*-
"""Functions common to input model parsers.

The core of this module is an abstract base class extracts an input model
written in some neural network library and prepares it for further processing in
the SNN toolbox.

.. autosummary::
    :nosignatures:

    AbstractModelParser

The idea is to make all further steps in the conversion/simulation pipeline
independent of the original model format.

Other functions help navigate through the network in order to explore network
connectivity and layer attributes:

.. autosummary::
    :nosignatures:

    get_type
    has_weights
    get_fanin
    get_fanout
    get_inbound_layers
    get_inbound_layers_with_params
    get_inbound_layers_without_params
    get_outbound_layers
    get_outbound_activation

@author: rbodo
"""

from abc import abstractmethod

import h5py
import keras
import numpy as np

from snntoolbox.utils.utils import binary_sigmoid, binary_tanh, binarize


class AbstractModelParser:
    """Abstract base class for neural network model parsers.

    Parameters
    ----------

    input_model
        The input network object.
    config: configparser.Configparser
        Contains the toolbox configuration for a particular experiment.

    Attributes
    ----------

    input_model: dict
        The input network object.
    config: configparser.Configparser
        Contains the toolbox configuration for a particular experiment.
    _layer_list: list[dict]
        A list where each entry is a dictionary containing layer
        specifications. Obtained by calling `parse`. Used to build new, parsed
        Keras model.
    _layer_dict: dict
        Maps the layer names of the specific input model library to our standard
        names (currently Keras).
    parsed_model: keras.models.Model
        The parsed model.
    """

    def __init__(self, input_model, config):
        self.input_model = input_model
        self.config = config
        self._layer_list = []
        self._layer_dict = {}
        self.parsed_model = None

    def parse(self):
        """Extract the essential information about a neural network.

        This method serves to abstract the conversion process of a network from
        the language the input model was built in (e.g. Keras or Lasagne).

        The methods iterates over all layers of the input model and writes the
        layer specifications and parameters into `_layer_list`. The keys are
        chosen in accordance with Keras layer attributes to facilitate
        instantiation of a new, parsed Keras model (done in a later step by
        `build_parsed_model`).

        This function applies several simplifications and adaptations to prepare
        the model for conversion to spiking. These modifications include:

        - Removing layers only used during training (Dropout,
          BatchNormalization, ...)
        - Absorbing the parameters of BatchNormalization layers into the
          parameters of the preceeding layer. This does not affect performance
          because batch-norm-parameters are constant at inference time.
        - Removing ReLU activation layers, because their function is inherent to
          the spike generation mechanism. The information which nonlinearity was
          used in the original model is preserved in the ``activation`` key in
          `_layer_list`. If the output layer employs the softmax function, a
          spiking version is used when testing the SNN in INIsim or MegaSim
          simulators.
        - Inserting a Flatten layer between Conv and FC layers, if the input
          model did not explicitly include one.
        """

        layers = self.get_layer_iterable()
        snn_layers = eval(self.config['restrictions']['snn_layers'])

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
                parameters = list(
                    self._layer_list[prev_layer_idx]['parameters'])
                print("Absorbing batch-normalization parameters into " +
                      "parameters of previous {}.".format(self.get_type(
                          prev_layer)))
                self._layer_list[prev_layer_idx]['parameters'] = \
                    absorb_bn_parameters(*(parameters + parameters_bn))

            if layer_type == 'GlobalAveragePooling2D':
                print("Replacing GlobalAveragePooling by AveragePooling "
                      "plus Flatten.")
                pool_size = [layer.input_shape[-2], layer.input_shape[-1]]
                self._layer_list.append(
                    {'layer_type': 'AveragePooling2D',
                     'name': self.get_name(layer, idx, 'AveragePooling2D'),
                     'input_shape': layer.input_shape, 'pool_size': pool_size,
                     'inbound': self.get_inbound_names(layer, name_map)})
                name_map['AveragePooling2D' + str(idx)] = idx
                idx += 1
                num_str = str(idx) if idx > 9 else '0' + str(idx)
                shape_string = str(np.prod(layer.output_shape[1:]))
                self._layer_list.append(
                    {'name': num_str + 'Flatten_' + shape_string,
                     'layer_type': 'Flatten',
                     'inbound': [self._layer_list[-1]['name']]})
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
                inbound = [self._layer_list[-1]['name']]
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

            self._layer_list.append(attributes)

            # Map layer index to layer id. Needed for inception modules.
            name_map[str(id(layer))] = idx

            idx += 1
        print('')

    @abstractmethod
    def get_layer_iterable(self):
        """Get an iterable over the layers of the network.

        Returns
        -------

        layers: list
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
        """Get the parameters of a batch-normalization layer.

        Returns
        -------

        mean, var_eps_sqrt_inv, gamma, beta, axis: tuple

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

        Parameters
        ----------

        layer:
            Layer
        name_map: dict
            Maps the name of a layer to the `id` of the layer object.

        Returns
        -------

        : list
            The names of inbound layers.

        """

        if len(self._layer_list) == 0:
            return ['input']

        inbound = self.get_inbound_layers(layer)
        for ib in range(len(inbound)):
            for _ in range(len(self.layers_to_skip)):
                if self.get_type(inbound[ib]) in self.layers_to_skip:
                    inbound[ib] = self.get_inbound_layers(inbound[ib])[0]
                else:
                    break
        inb_idxs = [name_map[str(id(inb))] for inb in inbound]
        return [self._layer_list[i]['name'] for i in inb_idxs]

    @abstractmethod
    def get_inbound_layers(self, layer):
        """Get inbound layers of ``layer``.

        Returns
        -------

        inbound: Sequence
        """

        pass

    @property
    def layers_to_skip(self):
        """
        Return a list of layer names that should be skipped during conversion
        to a spiking network.

        Returns
        -------

        self._layers_to_skip: List[str]
        """

        return ['BatchNormalization', 'Activation', 'Dropout']

    @abstractmethod
    def has_weights(self, layer):
        """Return ``True`` if ``layer`` has weights."""

        pass

    def initialize_attributes(self, layer=None):
        """
        Return a dictionary that will be used to collect all attributes of a
        layer. This dictionary can then be used to instantiate a new parsed
        layer.
        """

        return {}

    @abstractmethod
    def get_input_shape(self):
        """Get the input shape of a network, not including batch size.

        Returns
        -------

        input_shape: tuple
            Input shape.
        """

        pass

    def get_batch_input_shape(self):
        """Get the input shape of a network, including batch size.

        Returns
        -------

        batch_input_shape: tuple
            Batch input shape.
        """

        input_shape = tuple(self.get_input_shape())
        batch_size = self.config.getint('simulation', 'batch_size')
        return (batch_size,) + input_shape

    def get_name(self, layer, idx, layer_type=None):
        """Create a name for a ``layer``.

        The format is <layer_num><layer_type>_<layer_shape>.

        >>> # Name of first convolution layer with 32 feature maps and dimension
        >>> # 64x64:
        "00Conv2D_32x64x64"
        >>> # Name of final dense layer with 100 units:
        "06Dense_100"

        Parameters
        ----------

        layer:
            Layer.
        idx: int
            Layer index.
        layer_type: Optional[str]
            Type of layer.

        Returns
        -------

        name: str
            Layer name.
        """

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
        """Get output shape of a ``layer``.

        Parameters
        ----------

        layer
            Layer.

        Returns
        -------

        output_shape: Sized
            Output shape of ``layer``.
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
            self._layer_list.append({
                'name': num_str + 'Flatten_' + shape_string,
                'layer_type': 'Flatten',
                'inbound': self.get_inbound_names(layer, name_map)})
            name_map['Flatten' + str(idx)] = idx
            return True
        else:
            return False

    @abstractmethod
    def parse_dense(self, layer, attributes):
        """Parse a fully-connected layer.

        Parameters
        ----------

        layer:
            Layer.
        attributes: dict
            The layer attributes as key-value pairs in a dict.
        """

        pass

    @abstractmethod
    def parse_convolution(self, layer, attributes):
        """Parse a convolutional layer.

        Parameters
        ----------

        layer:
            Layer.
        attributes: dict
            The layer attributes as key-value pairs in a dict.
        """

        pass

    @abstractmethod
    def parse_pooling(self, layer, attributes):
        """Parse a pooling layer.

        Parameters
        ----------

        layer:
            Layer.
        attributes: dict
            The layer attributes as key-value pairs in a dict.
        """

        pass

    def absorb_activation(self, layer, attributes):
        """Detect what activation is used by the layer.

        Sometimes the Dense or Conv layer specifies its activation directly,
        sometimes it is followed by a dedicated Activation layer (possibly
        with BatchNormalization in between). Here we try to find such an
        activation layer, and add this information to the Dense/Conv layer
        itself. The separate Activation layer can then be removed.

        Parameters
        ----------

        layer:
            Layer.
        attributes: dict
            The layer attributes as key-value pairs in a dict.
        """

        activation = self.get_activation(layer)

        outbound = layer
        for _ in range(3):
            outbound = list(self.get_outbound_layers(outbound))
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
        """Get the activation string of an activation ``layer``.

        Parameters
        ----------

        layer
            Layer

        Returns
        -------

        activation: str
            String indicating the activation of the ``layer``.

        """

        pass

    @abstractmethod
    def get_outbound_layers(self, layer):
        """Get outbound layers of ``layer``.

        Parameters
        ----------

        layer:
            Layer.

        Returns
        -------

        outbound: list
            Outbound layers of ``layer``.

        """

        pass

    @abstractmethod
    def parse_concatenate(self, layer, attributes):
        """Parse a concatenation layer.

        Parameters
        ----------

        layer:
            Layer.
        attributes: dict
            The layer attributes as key-value pairs in a dict.
        """

        pass

    def build_parsed_model(self):
        """Create a Keras model suitable for conversion to SNN.

        This method uses the specifications in `_layer_list` to build a
        Keras model. The resulting model contains all essential information
        about the original network, independently of the model library in which
        the original network was built (e.g. Caffe).

        Returns
        -------

        parsed_model: keras.models.Model
            A Keras model, functionally equivalent to `input_model`.
        """

        img_input = keras.layers.Input(batch_shape=self.get_batch_input_shape(),
                                       name='input')
        parsed_layers = {'input': img_input}
        print("Building parsed model...\n")
        for layer in self._layer_list:
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
            self._layer_list[-1]['name']])
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
        (``keras.ImageDataGenerator.flow_from_directory`` object).

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
    if axis != -1:
        # print("Warning: Specifying a batch-normalization axis other than the "
        #       "default (-1) has not been thoroughly tested yet.")
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


def load_parameters(filepath):
    """Load all layer parameters from an HDF5 file."""

    f = h5py.File(filepath, 'r')

    params = []
    for k in sorted(f.keys()):
        params.append(np.array(f.get(k)))

    f.close()

    return params


def save_parameters(params, filepath, fileformat='h5'):
    """Save all layer parameters to an HDF5 file."""

    if fileformat == 'pkl':
        import pickle
        pickle.dump(params, open(filepath + '.pkl', str('wb')))
    else:
        with h5py.File(filepath, mode='w') as f:
            for i, p in enumerate(params):
                if i < 10:
                    j = '00' + str(i)
                elif i < 100:
                    j = '0' + str(i)
                else:
                    j = str(i)
                f.create_dataset('param_'+j, data=p)


def has_weights(layer):
    """Return ``True`` if layer has weights.

    Parameters
    ----------

    layer : keras.layers.Layer
        Keras layer

    Returns
    -------

    : bool
        ``True`` if layer has weights.
    """

    return len(layer.weights)


def get_inbound_layers_with_params(layer):
    """Iterate until inbound layers are found that have parameters.

    Parameters
    ----------

    layer: keras.layers.Layer
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
            if has_weights(inbound) > 0:
                return [inbound]
        else:
            result = []
            for inb in inbound:
                if has_weights(inb):
                    result.append(inb)
                else:
                    result += get_inbound_layers_with_params(inb)
            return result


def get_inbound_layers_without_params(layer):
    """Return inbound layers.

    Parameters
    ----------

    layer: Keras.layers
        A Keras layer.

    Returns
    -------

    : list[Keras.layers]
        List of inbound layers.
    """

    return [layer for layer in layer.inbound_nodes[0].inbound_layers
            if len(layer.weights) == 0]


def get_inbound_layers(layer):
    """Return inbound layers.

    Parameters
    ----------

    layer: Keras.layers
        A Keras layer.

    Returns
    -------

    : list[Keras.layers]
        List of inbound layers.
    """

    return layer.inbound_nodes[0].inbound_layers


def get_outbound_layers(layer):
    """Return outbound layers.

    Parameters
    ----------

    layer: Keras.layers
        A Keras layer.

    Returns
    -------

    : list[Keras.layers]
        List of outbound layers.
    """

    return [on.outbound_layer for on in layer.outbound_nodes]


def get_outbound_activation(layer):
    """
    Iterate over 2 outbound layers to find an activation layer. If there is no
    activation layer, take the activation of the current layer.

    Parameters
    ----------

    layer: Union[keras.layers.Conv2D, keras.layers.Dense]
        Layer

    Returns
    -------

    activation: str
        Name of outbound activation type.
    """

    activation = layer.activation.__name__
    outbound = layer
    for _ in range(2):
        outbound = get_outbound_layers(outbound)
        if len(outbound) == 1 and get_type(outbound[0]) == 'Activation':
            activation = outbound[0].activation.__name__
    return activation


def get_fanin(layer):
    """
    Return fan-in of a neuron in ``layer``.

    Parameters
    ----------

    layer: Subclass[keras.layers.Layer]
         Layer.

    Returns
    -------

    fanin: int
        Fan-in.

    """

    if 'Conv' in layer.name:
        fanin = np.prod(layer.kernel_size) * layer.input_shape[1]
    elif 'Dense' in layer.name:
        fanin = layer.input_shape[1]
    elif 'Pool' in layer.name:
        fanin = 0
    else:
        fanin = 0

    return fanin


def get_fanout(layer, config):
    """
    Return fan-out of a neuron in ``layer``.

    Parameters
    ----------

    layer: Subclass[keras.layers.Layer]
         Layer.
    config: configparser.ConfigParser
        Settings.

    Returns
    -------

    fanout: int
        Fan-out.

    """

    from snntoolbox.simulation.utils import get_spiking_outbound_layers

    fanout = 0
    next_layers = get_spiking_outbound_layers(layer, config)
    for next_layer in next_layers:
        if 'Conv' in layer.name and 'Pool' in next_layer.name:
            fanout += 1
        elif 'Dense' in layer.name:
            fanout += next_layer.units
        elif 'Pool' in layer.name and 'Conv' in next_layer.name:
            fanout += np.prod(next_layer.kernel_size) * next_layer.filters
        elif 'Pool' in layer.name and 'Dense' in next_layer.name:
            fanout += next_layer.units
        else:
            fanout += 0

    return fanout


def get_type(layer):
    """Get type of Keras layer.

    Parameters
    ----------

    layer: Keras.layers.Layer
        Keras layer.

    Returns
    -------

    : str
        Layer type.

    """

    return layer.__class__.__name__
