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

import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, AveragePooling2D, \
    MaxPooling2D, Conv2D, Layer, Concatenate, ZeroPadding2D, Reshape, \
    DepthwiseConv2D


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
        self._floatx = tf.keras.backend.floatx()

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
        if hasattr(self, 'activation_str') \
                and self.activation_str == 'softmax':
            output_spikes = self.softmax_activation(new_mem)
        else:
            output_spikes = self.linear_activation(new_mem)

        # Reset membrane potential after spikes.
        self.set_reset_mem(new_mem, output_spikes)

        # Store refractory period after spikes.
        if hasattr(self, 'activation_str') \
                and self.activation_str == 'softmax':
            # We do not constrain softmax output neurons.
            new_refrac = tf.identity(self.refrac_until)
        else:
            new_refrac = tf.where(tf.not_equal(output_spikes, 0),
                                  self.time + self.tau_refrac,
                                  self.refrac_until)
        self.refrac_until.assign(new_refrac)

        if self.spiketrain is not None:
            self.spiketrain.assign(self.time * tf.cast(
                tf.not_equal(output_spikes, 0), self._floatx))

        # Compute post-synaptic potential.
        psp = self.get_psp(output_spikes)

        return tf.cast(psp, self._floatx)

    def linear_activation(self, mem):
        """Linear activation."""
        return tf.cast(tf.greater_equal(mem, self.v_thresh), self._floatx)

    def softmax_activation(self, mem):
        """Softmax activation."""

        return tf.cast(tf.less_equal(tf.random.uniform(tf.shape(mem)),
                                     tf.nn.softmax(mem)), self._floatx)

    def get_new_mem(self):
        """Add input to membrane potential."""

        # Destroy impulse if in refractory period
        masked_impulse = self.impulse if self.tau_refrac == 0 else \
            tf.where(tf.greater(self.refrac_until, self.time),
                     tf.zeros_like(self.impulse), self.impulse)

        new_mem = self.mem + masked_impulse

        if self.config.getboolean('cell', 'leak'):
            # Todo: Implement more flexible version of leak!
            new_mem = tf.where(tf.greater(new_mem, 0),
                               new_mem - 0.1 * self.dt, new_mem)

        return new_mem

    def set_reset_mem(self, mem, spikes):
        """
        Reset membrane potential ``mem`` array where ``spikes`` array is
        nonzero.
        """

        if hasattr(self, 'activation_str') \
                and self.activation_str == 'softmax':
            new = tf.identity(mem)
        else:
            new = tf.where(tf.not_equal(spikes, 0), tf.zeros_like(mem), mem)
        self.mem.assign(new)

    def get_psp(self, output_spikes):
        if hasattr(self, 'activation_str') \
                and self.activation_str == 'softmax':
            psp = tf.identity(output_spikes)
        else:
            new_spiketimes = tf.where(tf.not_equal(output_spikes, 0),
                                      tf.ones_like(output_spikes) * self.time,
                                      self.last_spiketimes)
            assign_new_spiketimes = self.last_spiketimes.assign(new_spiketimes)
            with tf.control_dependencies([assign_new_spiketimes]):
                last_spiketimes = self.last_spiketimes + 0  # Dummy op
                psp = tf.where(tf.greater(last_spiketimes, 0),
                               tf.ones_like(output_spikes) * self.dt,
                               tf.zeros_like(output_spikes))
        return psp

    def get_time(self):
        """Get simulation time variable.

            Returns
            -------

            time: float
                Current simulation time.
            """

        return self.time.eval

    def set_time(self, time):
        """Set simulation time variable.

        Parameters
        ----------

        time: float
            Current simulation time.
        """

        self.time.assign(time)

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
            init_mem = tf.random.uniform(output_shape,
                                         -self._v_thresh, self._v_thresh)
        elif mode == 'bias':
            init_mem = tf.zeros(output_shape, self._floatx)
            if hasattr(self, 'bias'):
                bias = self.get_weights()[1]
                for i in range(len(bias)):
                    # Todo: This assumes data_format = 'channels_first'
                    init_mem[:, i, Ellipsis] = bias[i]
                self.bias.assign(tf.zeros_like(bias))
        else:  # mode == 'zero':
            init_mem = tf.zeros(output_shape, self._floatx)
        return init_mem

    @tf.function
    def reset_spikevars(self, sample_idx):
        """
        Reset variables present in spiking layers. Can be turned off for
        instance when a video sequence is tested.
        """

        mod = self.config.getint('simulation', 'reset_between_nth_sample')
        mod = mod if mod else sample_idx + 1
        do_reset = sample_idx % mod == 0
        if do_reset:
            self.mem.assign(self.init_membrane_potential())
        self.time.assign(self.dt)
        zeros_output_shape = tf.zeros(self.output_shape, self._floatx)
        if self.tau_refrac > 0:
            self.refrac_until.assign(zeros_output_shape)
        if self.spiketrain is not None:
            self.spiketrain.assign(zeros_output_shape)
        self.last_spiketimes.assign(zeros_output_shape - 1)

    @tf.function
    def init_neurons(self, input_shape):
        """Init layer neurons."""

        from snntoolbox.bin.utils import get_log_keys, get_plot_keys

        output_shape = self.compute_output_shape(input_shape)
        if self.v_thresh is None:
            self.v_thresh = tf.Variable(self._v_thresh, name='v_thresh',
                                        trainable=False)
        if self.mem is None:
            self.mem = tf.Variable(self.init_membrane_potential(output_shape),
                                   name='v_mem', trainable=False)
        if self.time is None:
            self.time = tf.Variable(self.dt, name='dt', trainable=False)
        # To save memory and computations, allocate only where needed:
        if self.tau_refrac > 0 and self.refrac_until is None:
            self.refrac_until = tf.Variable(
                tf.zeros(output_shape), name='refrac_until', trainable=False)
        if any({'spiketrains', 'spikerates', 'correlation', 'spikecounts',
                'hist_spikerates_activations', 'operations',
                'synaptic_operations_b_t', 'neuron_operations_b_t',
                'spiketrains_n_b_l_t'} & (get_plot_keys(self.config) |
               get_log_keys(self.config))) and self.spiketrain is None:
            self.spiketrain = tf.Variable(tf.zeros(output_shape),
                                          name='spiketrains', trainable=False)
        if self.last_spiketimes is None:
            self.last_spiketimes = tf.Variable(-tf.ones(output_shape),
                                               name='last_spiketimes',
                                               trainable=False)

    def get_layer_idx(self):
        """Get index of layer."""

        label = self.name.split('_')[0]
        layer_idx = None
        for i in range(len(label)):
            if label[:i].isdigit():
                layer_idx = int(label[:i])
        return layer_idx


def spike_call(call):
    @tf.function
    def decorator(self, x):

        # Only call layer if there are input spikes. This is to prevent
        # accumulation of bias.
        self.impulse = tf.cond(tf.math.reduce_any(tf.not_equal(x, 0)),
                               lambda: call(self, x),
                               lambda: tf.zeros_like(self.mem))
        return self.update_neurons()

    return decorator


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


class SpikeZeroPadding2D(ZeroPadding2D):
    """Spike padding layer"""

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

    @property
    def class_name(self):
        """Get class name."""

        return self.__class__.__name__


class SpikeReshape(Reshape):
    """Spike reshape layer"""

    def __init__(self, *args, **kwargs):
        kwargs.pop(str('config'))
        Reshape.__init__(self, *args, **kwargs)

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
        self.init_neurons(input_shape.as_list())

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
        self.init_neurons(input_shape.as_list())

    @spike_call
    def call(self, x, mask=None):

        return Conv2D.call(self, x)


class SpikeDepthwiseConv2D(DepthwiseConv2D, SpikeLayer):
    """Spike 2D depthwise-separable convolution."""

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
        self.init_neurons(input_shape.as_list())

    @spike_call
    def call(self, x, mask=None):

        return DepthwiseConv2D.call(self, x)


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
        self.init_neurons(input_shape.as_list())

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
        self.init_neurons(input_shape.as_list())

    def call(self, x, mask=None):
        """Layer functionality."""
        # Skip integration of input spikes in membrane potential. Directly
        # transmit new spikes. The output psp is nonzero wherever there has
        # been an input spike at any time during simulation.

        input_psp = MaxPooling2D.call(self, x)

        if self.spiketrain is not None:
            new_spikes = tf.math.logical_xor(
                tf.greater(input_psp, 0), tf.greater(self.last_spiketimes, 0))
            self.spiketrain.assign(self.time * tf.cast(new_spikes,
                                                       self._floatx))

        psp = self.get_psp(input_psp)

        return tf.cast(psp, self._floatx)


custom_layers = {'SpikeFlatten': SpikeFlatten,
                 'SpikeDense': SpikeDense,
                 'SpikeConv2D': SpikeConv2D,
                 'SpikeAveragePooling2D': SpikeAveragePooling2D,
                 'SpikeMaxPooling2D': SpikeMaxPooling2D,
                 'SpikeConcatenate': SpikeConcatenate,
                 'SpikeDepthwiseConv2D': SpikeDepthwiseConv2D,
                 'SpikeZeroPadding2D': SpikeZeroPadding2D,
                 'SpikeReshape': SpikeReshape}
