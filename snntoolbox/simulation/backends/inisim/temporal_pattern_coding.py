# -*- coding: utf-8 -*-
"""INI spiking neuron simulator.

This module defines the layer objects used to create a spiking neural network
for our built-in INI simulator
:py:mod:`~snntoolbox.simulation.target_simulators.INI_target_sim`.

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

# import tensorflow as tf
# from tensorflow.python import debug as tf_debug
# k.set_session(tf_debug.LocalCLIDebugWrapperSession(tf.Session()))

standard_library.install_aliases()


class SpikeLayer(Layer):
    """Base class for layer with spiking neurons."""

    def __init__(self, **kwargs):
        self.config = kwargs.pop(str('config'), None)
        self.layer_type = self.class_name
        self.spiketrain = self.spikerates = None

        finfo = np.finfo(self.config.get('conversion', 'activation_dtype'))
        self.scale_fac_inv = k.constant(min(finfo.epsneg, finfo.eps),
                                        k.floatx(), ())
        self.scale_fac = 1 / self.scale_fac_inv
        self.num_bits = finfo.bits

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

        from snntoolbox.bin.utils import get_log_keys, get_plot_keys

        output_shape = self.compute_output_shape(input_shape)
        if any({'spiketrains', 'spikerates', 'correlation', 'spikecounts',
                'hist_spikerates_activations', 'operations',
                'synaptic_operations_b_t', 'neuron_operations_b_t',
                'spiketrains_n_b_l_t'} & (get_plot_keys(self.config) |
               get_log_keys(self.config))):
            self.spiketrain = k.zeros(list(output_shape) + [self.num_bits])
            self.spikerates = k.zeros(output_shape)


def spike_call(call):
    def decorator(self, x):

        # Transform x into binary format here. Effective batch_size increases
        # from 1 to num_bits.
        x = to_binary(x, self.num_bits, self.scale_fac)
        # Multiply binary feature map matrix by PSP kernel which decays
        # exponentially across the 32 temporal steps (batch-dimension).
        shape = [self.num_bits] + [1] * len(x.shape[1:])
        x *= k.constant([2**(self.num_bits-i-1) for i in range(self.num_bits)], k.floatx(), shape)

        self.impulse = call(self, x)
        pre_activ = k.sum(self.impulse, 0, keepdims=True) * self.scale_fac_inv
        activ = softmax(pre_activ) if self.activation_str == 'softmax' \
            else relu(pre_activ)

        # if self.spiketrain is not None:
        #     y = to_binary(activ, self.num_bits, self.scale_fac)
        #     shape = [self.num_bits] + [1] * len(y.shape[1:])
        #     y *= k.reshape(k.arange(self.num_bits, dtype=k.floatx()), shape)
        #     shape_y = k.get_variable_shape(y)
        #     self.add_update([(self.spiketrain, k.reshape(
        #         y, (1,) + tuple(shape_y[1:]) + (shape_y[0],)))])

        if self.spikerates is not None:
            self.add_update([self.spikerates, activ])

        return activ

    return decorator


def to_binary(x, num_bits, scale_fac):
    """Transform an array of floats into binary representation.

    Parameters
    ----------

    x: ndarray
        Input array containing float values. The first dimension has to be of
        length 1.
    num_bits: int
        The fixed point precision to be used when converting to binary. Will be
        inferred from ``x`` if not specified.
    scale_fac: float
        Factor to scale from float to int. Because activations are normalized,
        we do not need to check for overflow when scaling the activations ``x``
        by ``scale_fac``. (Assumes that the inverse floatX.eps is smaller than
        intX.max.)

    Returns
    -------

    binary_array: ndarray
        Output boolean array. The first dimension of x is expanded to length
        ``bits``. The binary representation of each value in ``x`` is
        distributed across the first dimension of ``binary_array``.
    """

    shape = k.get_variable_shape(x)

    binary_array = k.zeros([num_bits] + list(shape[1:]), k.floatx())

    powers = k.constant([2**(num_bits-i-1) for i in range(num_bits)],
                        k.floatx(), (num_bits,))

    f = k.variable(0., k.floatx())
    # f = k.tf.Print(f, [f])

    def get_bit(ff, ii):
        k.update_sub(ff, powers[ii])
        return 1.

    if len(shape) == 2:
        for l in range(shape[1]):
            k.update(f, x[0, l] * scale_fac)
            for i in range(num_bits):
                b = k.tf.cond(k.greater_equal(f, powers[i]),
                              lambda: get_bit(f, i), lambda: 0.)
                k.update(binary_array[i, l], b)
    elif len(shape) == 4:
        import itertools
        for l, m, n in itertools.product(
                range(shape[1]), range(shape[2]), range(shape[3])):
            k.update(f, x[0, l, m, n] * scale_fac)
            for i in range(num_bits):
                b = k.tf.cond(k.greater_equal(f, powers[i]),
                              lambda: get_bit(f, i), lambda: 0.)
                k.update(binary_array[i, l, m, n], b)
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

    def call(self, x, mask=None):

        return k.cast(super(SpikeFlatten, self).call(x), k.floatx())

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


custom_layers = {'SpikeFlatten': SpikeFlatten,
                 'SpikeDense': SpikeDense,
                 'SpikeConv2D': SpikeConv2D,
                 'SpikeAveragePooling2D': SpikeAveragePooling2D,
                 'SpikeMaxPooling2D': SpikeMaxPooling2D,
                 'SpikeConcatenate': SpikeConcatenate}
