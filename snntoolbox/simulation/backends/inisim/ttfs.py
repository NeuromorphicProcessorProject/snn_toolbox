# -*- coding: utf-8 -*-
"""INI time-to-first-spike simulator backend.

This module defines the layer objects used to create a spiking neural network
for our built-in INI simulator
:py:mod:`~snntoolbox.simulation.target_simulators.INI_ttfs_target_sim`.

The coding scheme underlying this conversion is that the instantaneous firing
rate is given by the inverse time-to-first-spike.

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

standard_library.install_aliases()


class SpikeLayer(Layer):
    """Base class for layer with spiking neurons."""

    def __init__(self, **kwargs):
        self.config = kwargs.pop(str('config'), None)
        self.layer_type = self.class_name
        self.batch_size = self.config.getint('simulation', 'batch_size')
        self.dt = self.config.getfloat('simulation', 'dt')
        self.duration = self.config.getint('simulation', 'duration')
        self.tau_refrac = self.config.getfloat('cell', 'tau_refrac')
        self._v_thresh = self.config.getfloat('cell', 'v_thresh')
        self.v_thresh = None
        self.time = None
        self.mem = self.spiketrain = self.impulse = None
        self.refrac_until = None
        self.last_spiketimes = None

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

    def reset(self, sample_idx):
        """Reset layer variables."""

        self.reset_spikevars(sample_idx)

    @property
    def class_name(self):
        """Get class name."""

        return self.__class__.__name__

    def update_neurons(self):
        """Update neurons according to activation function."""

        # Update membrane potentials.
        new_mem = self.get_new_mem()

        # Generate spikes.
        if hasattr(self, 'activation_str') and self.activation_str == 'softmax':
            output_spikes = self.softmax_activation(new_mem)
        else:
            output_spikes = self.linear_activation(new_mem)

        # Reset membrane potential after spikes.
        self.set_reset_mem(new_mem, output_spikes)

        # Store refractory period after spikes.
        if hasattr(self, 'activation_str') and self.activation_str == 'softmax':
            # We do not constrain softmax output neurons.
            new_refrac = k.tf.identity(self.refrac_until)
        else:
            new_refrac = k.tf.where(k.not_equal(output_spikes, 0),
                                    k.ones_like(output_spikes) *
                                    (self.time + self.tau_refrac),
                                    self.refrac_until)
        self.add_update([(self.refrac_until, new_refrac)])

        if self.spiketrain is not None:
            self.add_update([(self.spiketrain, self.time * k.cast(
                k.not_equal(output_spikes, 0), k.floatx()))])

        # Compute post-synaptic potential.
        psp = self.get_psp(output_spikes)

        return k.cast(psp, k.floatx())

    def linear_activation(self, mem):
        """Linear activation."""
        return k.cast(k.greater_equal(mem, self.v_thresh), k.floatx())

    def softmax_activation(self, mem):
        """Softmax activation."""

        return k.cast(k.less_equal(k.random_uniform(k.shape(mem)),
                                   k.softmax(mem)), k.floatx())

    def get_new_mem(self):
        """Add input to membrane potential."""

        # Destroy impulse if in refractory period
        masked_impulse = self.impulse if self.tau_refrac == 0 else \
            k.tf.where(k.greater(self.refrac_until, self.time),
                       k.zeros_like(self.impulse), self.impulse)

        new_mem = self.mem + masked_impulse

        if self.config.getboolean('cell', 'leak'):
            # Todo: Implement more flexible version of leak!
            new_mem = k.tf.where(k.greater(new_mem, 0), new_mem - 0.1 * self.dt,
                                 new_mem)

        return new_mem

    def set_reset_mem(self, mem, spikes):
        """
        Reset membrane potential ``mem`` array where ``spikes`` array is
        nonzero.
        """

        if hasattr(self, 'activation_str') and self.activation_str == 'softmax':
            new = k.tf.identity(mem)
        else:
            new = k.tf.where(k.not_equal(spikes, 0), k.zeros_like(mem), mem)
        self.add_update([(self.mem, new)])

    def get_psp(self, output_spikes):
        if hasattr(self, 'activation_str') and self.activation_str == 'softmax':
            psp = k.tf.identity(output_spikes)
        else:
            new_spiketimes = k.tf.where(k.not_equal(output_spikes, 0),
                                        k.ones_like(output_spikes) * self.time,
                                        self.last_spiketimes)
            assign_new_spiketimes = k.tf.assign(self.last_spiketimes,
                                                new_spiketimes)
            with k.tf.control_dependencies([assign_new_spiketimes]):
                last_spiketimes = self.last_spiketimes + 0  # Dummy op
                psp = k.tf.where(k.greater(last_spiketimes, 0),
                                 k.ones_like(output_spikes) * self.dt,
                                 k.zeros_like(output_spikes))
        return psp

    def get_time(self):
        """Get simulation time variable.

            Returns
            -------

            time: float
                Current simulation time.
            """

        return k.get_value(self.time)

    def set_time(self, time):
        """Set simulation time variable.

        Parameters
        ----------

        time: float
            Current simulation time.
        """

        k.set_value(self.time, time)

    def init_membrane_potential(self, output_shape=None, mode='zero'):
        """Initialize membrane potential.

        Helpful to avoid transient response in the beginning of the simulation.
        Not needed when reset between frames is turned off, e.g. with a video
        data set.

        Parameters
        ----------

        output_shape: Optional[tuple]
            Output shape
        mode: str
            Initialization mode.

            - ``'uniform'``: Random numbers from uniform distribution in
              ``[-thr, thr]``.
            - ``'bias'``: Negative bias.
            - ``'zero'``: Zero (default).

        Returns
        -------

        init_mem: ndarray
            A tensor of ``self.output_shape`` (same as layer).
        """

        if output_shape is None:
            output_shape = self.output_shape

        if mode == 'uniform':
            init_mem = k.random_uniform(output_shape,
                                        -self._v_thresh, self._v_thresh)
        elif mode == 'bias':
            init_mem = np.zeros(output_shape, k.floatx())
            if hasattr(self, 'bias'):
                bias = self.get_weights()[1]
                for i in range(len(bias)):
                    # Todo: This assumes data_format = 'channels_first'
                    init_mem[:, i, Ellipsis] = bias[i]
                self.add_update([(self.bias, np.zeros_like(bias))])
        else:  # mode == 'zero':
            init_mem = np.zeros(output_shape, k.floatx())
        return init_mem

    def reset_spikevars(self, sample_idx):
        """
        Reset variables present in spiking layers. Can be turned off for
        instance when a video sequence is tested.
        """

        mod = self.config.getint('simulation', 'reset_between_nth_sample')
        mod = mod if mod else sample_idx + 1
        do_reset = sample_idx % mod == 0
        if do_reset:
            k.set_value(self.mem, self.init_membrane_potential())
        k.set_value(self.time, np.float32(self.dt))
        zeros_output_shape = np.zeros(self.output_shape, k.floatx())
        if self.tau_refrac > 0:
            k.set_value(self.refrac_until, zeros_output_shape)
        if self.spiketrain is not None:
            k.set_value(self.spiketrain, zeros_output_shape)
        k.set_value(self.last_spiketimes, zeros_output_shape - 1)

    def init_neurons(self, input_shape):
        """Init layer neurons."""

        from snntoolbox.bin.utils import get_log_keys, get_plot_keys

        output_shape = self.compute_output_shape(input_shape)
        self.v_thresh = k.variable(self._v_thresh)
        self.mem = k.variable(self.init_membrane_potential(output_shape))
        self.time = k.variable(self.dt)
        # To save memory and computations, allocate only where needed:
        if self.tau_refrac > 0:
            self.refrac_until = k.zeros(output_shape)
        if any({'spiketrains', 'spikerates', 'correlation', 'spikecounts',
                'hist_spikerates_activations', 'operations',
                'synaptic_operations_b_t', 'neuron_operations_b_t',
                'spiketrains_n_b_l_t'} & (get_plot_keys(self.config) |
               get_log_keys(self.config))):
            self.spiketrain = k.zeros(output_shape)
        self.last_spiketimes = k.variable(-np.ones(output_shape))

    def get_layer_idx(self):
        """Get index of layer."""

        label = self.name.split('_')[0]
        layer_idx = None
        for i in range(len(label)):
            if label[:i].isdigit():
                layer_idx = int(label[:i])
        return layer_idx


def spike_call(call):
    def decorator(self, x):

        # Only call layer if there are input spikes. This is to prevent
        # accumulation of bias.
        self.impulse = k.tf.cond(k.any(k.not_equal(x, 0)),
                                 lambda: call(self, x),
                                 lambda: k.zeros_like(self.mem))
        return self.update_neurons()

    return decorator


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
        self.config = kwargs.pop(str('config'), None)
        Flatten.__init__(self, **kwargs)

    @staticmethod
    def get_time():
        return None

    def reset(self, sample_idx):
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
    """Average Pooling."""

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

    @spike_call
    def call(self, x, mask=None):

        return AveragePooling2D.call(self, x)


class SpikeMaxPooling2D(MaxPooling2D, SpikeLayer):
    """Spiking Max Pooling."""

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
        """Layer functionality."""
        # Skip integration of input spikes in membrane potential. Directly
        # transmit new spikes. The output psp is nonzero wherever there has been
        # an input spike at any time during simulation.

        input_psp = MaxPooling2D.call(self, x)

        if self.spiketrain is not None:
            new_spikes = k.tf.logical_xor(k.greater(input_psp, 0),
                                          k.greater(self.last_spiketimes, 0))
            self.add_update([(self.spiketrain,
                              self.time * k.cast(new_spikes, k.floatx()))])

        psp = self.get_psp(input_psp)

        return k.cast(psp, k.floatx())


custom_layers = {'SpikeFlatten': SpikeFlatten,
                 'SpikeDense': SpikeDense,
                 'SpikeConv2D': SpikeConv2D,
                 'SpikeAveragePooling2D': SpikeAveragePooling2D,
                 'SpikeMaxPooling2D': SpikeMaxPooling2D,
                 'SpikeConcatenate': SpikeConcatenate}
