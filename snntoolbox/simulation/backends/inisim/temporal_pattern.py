# -*- coding: utf-8 -*-
"""INI temporal pattern simulator backend.

This module defines the layer objects used to create a spiking neural network
for our built-in INI simulator
:py:mod:`~snntoolbox.simulation.target_simulators.INI_temporal_pattern_target_sim`.

The coding scheme underlying this conversion is that the analog activation value
is transformed into a binary representation of spikes.

This simulator works only with Keras backend set to Tensorflow.

@author: rbodo
"""

from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import numpy as np
from future import standard_library
from keras import backend as k
from keras.layers import Dense, Flatten, AveragePooling2D, MaxPooling2D, Conv2D
from keras.layers import Layer, Concatenate
from keras.activations import softmax, relu

standard_library.install_aliases()


class SpikeLayer(Layer):
    """Base class for layer with spiking neurons."""

    def __init__(self, **kwargs):
        self.config = kwargs.pop(str('config'), None)
        self.layer_type = self.class_name
        self.spikerates = None
        self.num_bits = self.config.getint('conversion', 'num_bits')

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
        self.spikerates = k.zeros(output_shape)

    def update_spikevars(self, x):
        return [k.tf.assign(self.spikerates, x)]


def spike_call(call):
    def decorator(self, x):
        # Transform x into binary format here. Effective batch_size increases
        # from 1 to num_bits.
        x_binary = to_binary(x, self.num_bits)

        # Multiply binary feature map matrix by PSP kernel which decays
        # exponentially across the 32 temporal steps (batch-dimension).
        shape = [self.num_bits] + [1] * len(x.shape[1:])
        x_powers = x_binary * k.constant(
            [2**-i for i in range(self.num_bits)], k.floatx(), shape)
        x_weighted = call(self, x_powers)
        x_preactiv = k.sum(x_weighted, 0, keepdims=True)
        x_activ = softmax(x_preactiv) if self.activation_str == 'softmax' \
            else relu(x_preactiv)

        updates = self.update_spikevars(x_activ)

        with k.tf.control_dependencies(updates):
            return x_activ + 0

    return decorator


def to_binary(x, num_bits):
    """Transform an array of floats into binary representation.

    Parameters
    ----------

    x: ndarray
        Input array containing float values. The first dimension has to be of
        length 1.
    num_bits: int
        The fixed point precision to be used when converting to binary.

    Returns
    -------

    binary_array: ndarray
        Output boolean array. The first dimension of x is expanded to length
        ``bits``. The binary representation of each value in ``x`` is
        distributed across the first dimension of ``binary_array``.
    """

    shape = k.get_variable_shape(x)

    binary_array = k.zeros([num_bits] + list(shape[1:]), k.floatx())

    powers = k.constant(
        [2**-i for i in range(num_bits)], k.floatx(), (num_bits,))
    idx_p0 = k.constant(0, 'int32')

    if len(shape) > 2:
        idx_l0 = k.constant(0, 'int32')
        idx_m0 = k.constant(0, 'int32')
        idx_n0 = k.constant(0, 'int32')

        # noinspection PyUnusedLocal
        def is_iterate_powers(act_value, idx_p, idx_l, idx_m, idx_n):
            return k.less(idx_p, num_bits)

        # noinspection PyUnusedLocal
        def is_iterate_neurons_l(idx_p, idx_l, idx_m, idx_n):
            return k.less(idx_l, shape[1])

        # noinspection PyUnusedLocal
        def is_iterate_neurons_m(idx_p, idx_l, idx_m, idx_n):
            return k.less(idx_m, shape[2])

        # noinspection PyUnusedLocal
        def is_iterate_neurons_n(idx_p, idx_l, idx_m, idx_n):
            return k.less(idx_n, shape[3])

        def iterate_neurons_l(idx_p, idx_l, idx_m, idx_n):
            idx_p, idx_l, idx_m, idx_n = k.tf.while_loop(
                is_iterate_neurons_m, iterate_neurons_m,
                [idx_p, idx_l, idx_m, idx_n])
            return idx_p, idx_l + 1, 0, idx_n

        def iterate_neurons_m(idx_p, idx_l, idx_m, idx_n):
            idx_p, idx_l, idx_m, idx_n = k.tf.while_loop(
                is_iterate_neurons_n, iterate_neurons_n,
                [idx_p, idx_l, idx_m, idx_n])
            return idx_p, idx_l, idx_m + 1, 0

        def iterate_neurons_n(idx_p, idx_l, idx_m, idx_n):
            act_value = x[0, idx_l, idx_m, idx_n]
            act_value, idx_p, idx_l, idx_m, idx_n = k.tf.while_loop(
                is_iterate_powers, iterate_powers,
                [act_value, idx_p, idx_l, idx_m, idx_n])
            with k.tf.control_dependencies(
                    [idx_p, act_value, idx_l, idx_m, idx_n]):
                return 0, idx_l, idx_m, idx_n + 1

        def iterate_powers(act_value, idx_p, idx_l, idx_m, idx_n):
            p = powers[idx_p]
            c = k.greater_equal(act_value, p)
            b = k.tf.cond(c, lambda: 1., lambda: 0.)
            a = k.tf.assign(binary_array[idx_p, idx_l, idx_m, idx_n], b)
            new_act_value = k.tf.cond(c, lambda: act_value - p,
                                      lambda: act_value)
            with k.tf.control_dependencies([a]):
                return new_act_value, idx_p + 1, idx_l, idx_m, idx_n

        idx_p_, idx_l_, idx_m_, idx_n_ = k.tf.while_loop(
            is_iterate_neurons_l, iterate_neurons_l,
            [idx_p0, idx_l0, idx_m0, idx_n0])
        with k.tf.control_dependencies([idx_p_, idx_l_, idx_m_, idx_n_]):
            return binary_array + 0
    else:
        idx_l0 = k.constant(0, 'int32')

        # noinspection PyUnusedLocal
        def is_iterate_neurons_l(idx_p, idx_l):
            return k.less(idx_l, shape[1])

        # noinspection PyUnusedLocal
        def is_iterate_powers(act_value, idx_p, idx_l):
            return k.less(idx_p, num_bits)

        def iterate_neurons_l(idx_p, idx_l):
            act_value = x[0, idx_l]
            act_value, idx_p, idx_l = k.tf.while_loop(
                is_iterate_powers, iterate_powers,
                [act_value, idx_p, idx_l])
            return 0, idx_l + 1

        def iterate_powers(act_value, idx_p, idx_l):
            p = powers[idx_p]
            c = k.greater_equal(act_value, p)
            b = k.tf.cond(c, lambda: 1., lambda: 0.)
            a = k.tf.assign(binary_array[idx_p, idx_l], b)
            new_act_value = k.tf.cond(c, lambda: act_value - p,
                                      lambda: act_value)
            with k.tf.control_dependencies([a]):
                return new_act_value, idx_p + 1, idx_l

        idx_p_, idx_l_ = k.tf.while_loop(
            is_iterate_neurons_l, iterate_neurons_l, [idx_p0, idx_l0])

        with k.tf.control_dependencies([idx_p_, idx_l_]):
            return binary_array + 0


def to_binary_numpy(x, num_bits):
    """Transform an array of floats into binary representation.

    Parameters
    ----------

    x: ndarray
        Input array containing float values. The first dimension has to be of
        length 1.
    num_bits: int
        The fixed point precision to be used when converting to binary.

    Returns
    -------

    binary_array: ndarray
        Output boolean array. The first dimension of x is expanded to length
        ``bits``. The binary representation of each value in ``x`` is
        distributed across the first dimension of ``binary_array``.
    """

    binary_array = np.zeros([num_bits] + list(x.shape[1:]))

    powers = [2**-i for i in range(num_bits)]

    if len(x.shape) > 2:
        for l in range(x.shape[1]):
            for m in range(x.shape[2]):
                for n in range(x.shape[3]):
                    f = x[0, l, m, n]
                    for i in range(num_bits):
                        if f >= powers[i]:
                            binary_array[i, l, m, n] = 1
                            f -= powers[i]
    else:
        for l in range(x.shape[1]):
            f = x[0, l]
            for i in range(num_bits):
                if f >= powers[i]:
                    binary_array[i, l] = 1
                    f -= powers[i]
    return binary_array


class SpikeConcatenate(Concatenate):
    """Spike merge layer"""

    def __init__(self, axis, **kwargs):
        kwargs.pop(str('config'))
        Concatenate.__init__(self, axis, **kwargs)

    def _merge_function(self, inputs):

        return self._merge_function(inputs)

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

    @spike_call
    def call(self, x, **kwargs):

        return Dense.call(self, x)


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

    @spike_call
    def call(self, x, mask=None):

        return Conv2D.call(self, x)


class SpikeAveragePooling2D(AveragePooling2D, SpikeLayer):
    """Spike Average Pooling layer."""

    def __init__(self, **kwargs):
        AveragePooling2D.__init__(self, **kwargs)

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

        updates = self.update_spikevars(activ)

        with k.tf.control_dependencies(updates):
            return activ + 0


class SpikeMaxPooling2D(MaxPooling2D, SpikeLayer):
    """Spike Max Pooling."""

    def __init__(self, **kwargs):
        MaxPooling2D.__init__(self, **kwargs)

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

        updates = self.update_spikevars(activ)

        with k.tf.control_dependencies(updates):
            return activ + 0


custom_layers = {'SpikeFlatten': SpikeFlatten,
                 'SpikeDense': SpikeDense,
                 'SpikeConv2D': SpikeConv2D,
                 'SpikeAveragePooling2D': SpikeAveragePooling2D,
                 'SpikeMaxPooling2D': SpikeMaxPooling2D,
                 'SpikeConcatenate': SpikeConcatenate}
