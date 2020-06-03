# -*- coding: utf-8 -*-
"""INI temporal pattern simulator backend.

This module defines the layer objects used to create a spiking neural network
for our built-in INI simulator
:py:mod:`~snntoolbox.simulation.target_simulators.INI_temporal_pattern_target_sim`.

The coding scheme underlying this conversion is that the analog activation
value is transformed into a binary representation of spikes.

This simulator works only with Keras backend set to Tensorflow.

@author: rbodo
"""

import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, AveragePooling2D, Layer, \
    MaxPooling2D, Conv2D, Concatenate, DepthwiseConv2D, Reshape, ZeroPadding2D


class SpikeLayer(Layer):
    """Base class for layer with spiking neurons."""

    def __init__(self, **kwargs):
        self.config = kwargs.pop(str('config'), None)
        self.layer_type = self.class_name
        self.spikerates = None
        self.num_bits = self.config.getint('conversion', 'num_bits')
        self.powers = tf.constant([2**-(i+1) for i in range(self.num_bits)])
        self._x_binary = None
        self._a = None
        allowed_kwargs = {'input_shape',
                          'batch_input_shape',
                          'batch_size',
                          'dtype',
                          'name',
                          'trainable',
                          'weights',
                          'input_dtype',  # legacy
                          }
        for kwarg in kwargs.copy():
            if kwarg not in allowed_kwargs:
                kwargs.pop(kwarg)
        Layer.__init__(self, **kwargs)
        self.stateful = True

    @property
    def class_name(self):
        """Get class name."""

        return self.__class__.__name__

    def get_time(self):

        pass

    @staticmethod
    def reset(sample_idx):
        """Reset layer variables."""

        pass

    def init_neurons(self, input_shape):
        """Init layer neurons."""

        output_shape = self.compute_output_shape(input_shape)
        self.spikerates = tf.Variable(tf.zeros(output_shape), trainable=False,
                                      name='spikerates')

    @tf.function
    def spike_call(self, x, call):

        # Allocate variable in which to place binary version of x.
        if self._x_binary is None:
            shape = [self.num_bits] + x.shape[1:].as_list()
            self._x_binary = tf.Variable(
                lambda: tf.zeros(shape, tf.keras.backend.floatx()),
                name='x_binary', trainable=False)
            self._a = tf.Variable(lambda: tf.zeros_like(x), name='activation',
                                  trainable=False)

        # If not using ReLU, some x values could be negative.
        # Remove and store signs to apply after binarization.
        signs = tf.sign(x)
        x = tf.abs(x)

        # Make sure input is normalized before binarization. Hidden layers are
        # normalized during parsing.
        if self.is_first_spiking:
            x_max = tf.reduce_max(x)
            x = tf.divide(x, x_max)
        else:
            x_max = 1

        # Transform x into binary format here. Effective batch_size increases
        # from 1 to num_bits.
        x = self.to_binary(x)

        # Apply signs and rescale back to original range.
        x = tf.multiply(x, signs * x_max)

        # Perform layer operation, e.g. convolution, on every power of 2.
        y = call(self, x)

        # Add up the weighted powers of 2 to recover the activation values.
        y = tf.reduce_sum(y, 0, keepdims=True)

        # Apply non-linearity.
        if self.activation_str == 'softmax':
            y = tf.nn.softmax(y)
        elif self.activation_str == 'relu':
            y = tf.nn.relu(y)

        self.spikerates.assign(y)

        return y

    @tf.function
    def to_binary(self, x):
        """Transform an array of floats into binary representation.

        Parameters
        ----------

        x: tf.Tensor
            Input tensor containing float values. The first dimension has to be
            of length 1.

        Returns
        -------

        x_binary: tf.Variable
            Output boolean array. The first dimension of ``x`` is expanded to
            length ``num_bits``. The binary representation of each value in
            ``x`` is distributed across the first dimension of ``x_binary``.
        """

        n = 2 ** self.num_bits - 1
        self._a.assign(tf.divide(tf.round(tf.multiply(x, n)), n))

        for i in tf.range(self.num_bits):
            mask = tf.cast(tf.greater(self._a, self.powers[i]), tf.float32)
            # Multiply binary feature map matrix by PSP kernel which decays
            # exponentially across the 32 temporal steps (batch-dimension).
            b = mask * self.powers[i]
            self._x_binary[i:i+1].assign(b)
            self._a.assign_sub(b)

        return self._x_binary


class SpikeConcatenate(Concatenate):
    """Spike merge layer"""

    def __init__(self, axis, **kwargs):
        kwargs.pop(str('config'))
        Concatenate.__init__(self, axis, **kwargs)

    @staticmethod
    def get_time():

        pass

    @staticmethod
    def reset(sample_idx):
        """Reset layer variables."""

        pass

    @property
    def class_name(self):
        """Get class name."""

        return self.__class__.__name__


class SpikeFlatten(Flatten):
    """Spike flatten layer."""

    def __init__(self, **kwargs):
        kwargs.pop(str('config'))
        Flatten.__init__(self, **kwargs)

    @staticmethod
    def get_time():

        pass

    @staticmethod
    def reset(sample_idx):
        """Reset layer variables."""

        pass

    @property
    def class_name(self):
        """Get class name."""

        return self.__class__.__name__


class SpikeReshape(Reshape):
    """Spike reshape layer."""

    def __init__(self, target_shape, **kwargs):
        kwargs.pop(str('config'))
        Reshape.__init__(self, target_shape, **kwargs)

    @staticmethod
    def get_time():

        pass

    @staticmethod
    def reset(sample_idx):
        """Reset layer variables."""

        pass


class SpikeZeroPadding2D(ZeroPadding2D):
    """Spike padding layer."""

    def __init__(self, *args, **kwargs):
        kwargs.pop(str('config'))
        ZeroPadding2D.__init__(self, *args, **kwargs)

    @staticmethod
    def get_time():

        pass

    @staticmethod
    def reset(sample_idx):
        """Reset layer variables."""

        pass


class SpikeDense(Dense, SpikeLayer):
    """Spike Dense layer."""

    def build(self, input_shape):
        """Creates the layer neurons and connections.

        Parameters
        ----------

        input_shape: Union[list, tuple, Any]
            Keras tensor (future input to layer) or list/tuple of Keras tensors
            to reference for weight shape computations.
        """

        Dense.build(self, input_shape)
        self.init_neurons(input_shape)

    def call(self, x, **kwargs):

        return self.spike_call(x, Dense.call)


class SpikeConv2D(Conv2D, SpikeLayer):
    """Spike 2D Convolution."""

    def build(self, input_shape):
        """Creates the layer weights.
        Must be implemented on all layers that have weights.

        Parameters
        ----------

        input_shape: Union[list, tuple, Any]
            Keras tensor (future input to layer) or list/tuple of Keras tensors
            to reference for weight shape computations.
        """

        Conv2D.build(self, input_shape)
        self.init_neurons(input_shape)

    def call(self, x, mask=None):

        return self.spike_call(x, Conv2D.call)


class SpikeDepthwiseConv2D(DepthwiseConv2D, SpikeLayer):
    """Spike 2D depthwise-separable Convolution."""

    def build(self, input_shape):
        """Creates the layer weights.
        Must be implemented on all layers that have weights.

        Parameters
        ----------

        input_shape: Union[list, tuple, Any]
            Keras tensor (future input to layer) or list/tuple of Keras tensors
            to reference for weight shape computations.
        """

        DepthwiseConv2D.build(self, input_shape)
        self.init_neurons(input_shape)

    def call(self, x, mask=None):

        return self.spike_call(x, DepthwiseConv2D.call)


class SpikeAveragePooling2D(AveragePooling2D, SpikeLayer):
    """Spike Average Pooling layer."""

    def build(self, input_shape):
        """Creates the layer weights.
        Must be implemented on all layers that have weights.

        Parameters
        ----------

        input_shape: Union[list, tuple, Any]
            Keras tensor (future input to layer) or list/tuple of Keras tensors
            to reference for weight shape computations.
        """

        AveragePooling2D.build(self, input_shape)
        self.init_neurons(input_shape)

    def call(self, x, mask=None):
        activ = AveragePooling2D.call(self, x)
        self.spikerates.assign(activ)
        return activ


class SpikeMaxPooling2D(MaxPooling2D, SpikeLayer):
    """Spike Max Pooling."""

    def build(self, input_shape):
        """Creates the layer neurons and connections..

        Parameters
        ----------

        input_shape: Union[list, tuple, Any]
            Keras tensor (future input to layer) or list/tuple of Keras tensors
            to reference for weight shape computations.
        """

        MaxPooling2D.build(self, input_shape)
        self.init_neurons(input_shape)

    def call(self, x, mask=None):
        activ = MaxPooling2D.call(self, x)
        self.spikerates.assign(activ)
        return activ


custom_layers = {'SpikeFlatten': SpikeFlatten,
                 'SpikeDense': SpikeDense,
                 'SpikeConv2D': SpikeConv2D,
                 'SpikeAveragePooling2D': SpikeAveragePooling2D,
                 'SpikeMaxPooling2D': SpikeMaxPooling2D,
                 'SpikeConcatenate': SpikeConcatenate,
                 'SpikeDepthwiseConv2D': SpikeDepthwiseConv2D,
                 'SpikeZeroPadding2D': SpikeZeroPadding2D,
                 'SpikeReshape': SpikeReshape}
