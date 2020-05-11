# -*- coding: utf-8 -*-
"""INI temporal mean rate simulator with Tensorflow backend.

This module defines the layer objects used to create a spiking neural network
for our built-in INI simulator
:py:mod:`~snntoolbox.simulation.target_simulators.INI_temporal_mean_rate_target_sim`.

The coding scheme underlying this conversion is that the analog activation
value is represented by the average over number of spikes that occur during the
simulation duration.

@author: rbodo
"""
import os

import json

import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, AveragePooling2D, \
    MaxPooling2D, Conv2D, DepthwiseConv2D, ZeroPadding2D, Reshape, Layer, \
    Concatenate

from snntoolbox.parsing.utils import get_inbound_layers

# Experimental
clamp_var = False
v_clip = False


class SpikeLayer(Layer):
    """Base class for layer with spiking neurons."""

    def __init__(self, **kwargs):
        self.config = kwargs.pop(str('config'), None)
        self.layer_type = self.class_name
        self.dt = self.config.getfloat('simulation', 'dt')
        self.duration = self.config.getint('simulation', 'duration')
        self.tau_refrac = self.config.getfloat('cell', 'tau_refrac')
        self._v_thresh = self.config.getfloat('cell', 'v_thresh')
        self.v_thresh = None
        self.time = None
        self.mem = self.spiketrain = self.impulse = self.spikecounts = None
        self.refrac_until = self.max_spikerate = None
        if clamp_var:
            self.spikerate = self.var = None

        from snntoolbox.utils.utils import get_abs_path
        path, filename = \
            get_abs_path(self.config.get('paths', 'filename_clamp_indices'),
                         self.config)
        if filename != '':
            filepath = os.path.join(path, filename)
            assert os.path.isfile(filepath), \
                "File with clamp indices not found at {}.".format(filepath)
            self.filename_clamp_indices = filepath
            self.clamp_idx = None

        self.payloads = None
        self.payloads_sum = None
        self.online_normalization = self.config.getboolean(
            'normalization', 'online_normalization')

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

        self.reset_spikevars(tf.constant(sample_idx))

    @property
    def class_name(self):
        """Get class name."""

        return self.__class__.__name__

    def update_neurons(self):
        """Update neurons according to activation function."""

        new_mem = self.get_new_mem()

        if hasattr(self, 'activation_str'):
            if self.activation_str == 'softmax':
                output_spikes = self.softmax_activation(new_mem)
            elif self.activation_str == 'binary_sigmoid':
                output_spikes = self.binary_sigmoid_activation(new_mem)
            elif self.activation_str == 'binary_tanh':
                output_spikes = self.binary_tanh_activation(new_mem)
            elif '_Q' in self.activation_str:
                m, f = map(int, self.activation_str[
                           self.activation_str.index('_Q') + 2:].split('.'))
                output_spikes = self.quantized_activation(new_mem, m, f)
            else:
                output_spikes = self.linear_activation(new_mem)
        else:
            output_spikes = self.linear_activation(new_mem)

        # Store spiking
        self.set_reset_mem(new_mem, output_spikes)

        # Store refractory
        if self.tau_refrac > 0:
            new_refractory = tf.where(tf.not_equal(output_spikes, 0),
                                      self.time + self.tau_refrac,
                                      self.refrac_until)
            self.refrac_until.assign(new_refractory)

        if self.payloads:
            residuals = tf.where(tf.not_equal(output_spikes, 0),
                                 new_mem - self._v_thresh, new_mem)
            self.update_payload(residuals, output_spikes)

        if self.online_normalization:
            self.spikecounts.assign_add(tf.cast(tf.not_equal(output_spikes, 0),
                                                self._floatx))
            self.max_spikerate.assign(tf.reduce_max(self.spikecounts)
                                      * self.dt / self.time)

        if self.spiketrain is not None:
            self.spiketrain.assign(tf.cast(tf.not_equal(output_spikes, 0),
                                           self._floatx) * self.time)

        return tf.cast(output_spikes, self._floatx)

    def update_payload(self, residuals, spikes):
        """Update payloads.

        Uses the residual of the membrane potential after spike.
        """

        idxs = tf.not_equal(spikes, 0)
        payloads = tf.where(idxs, residuals[idxs] - self.payloads_sum[idxs],
                            self.payloads)
        payloads_sum = tf.where(idxs, self.payloads_sum + self.payloads,
                                self.payloads_sum)
        self.payloads.assign(payloads)
        self.payloads_sum.assign(payloads_sum)

    def linear_activation(self, mem):
        """Linear activation."""
        return tf.cast(tf.greater_equal(mem, self.v_thresh), self._floatx) * \
            self.v_thresh

    def binary_sigmoid_activation(self, mem):
        """Binary sigmoid activation."""

        return tf.cast(tf.greater(mem, 0), self._floatx) * self.v_thresh

    def binary_tanh_activation(self, mem):
        """Binary tanh activation."""

        output_spikes = tf.cast(tf.greater(mem, 0), self._floatx) \
            * self.v_thresh
        output_spikes += tf.cast(tf.less(mem, 0), self._floatx) \
            * -self.v_thresh

        return output_spikes

    def softmax_activation(self, mem):
        """Softmax activation."""

        # spiking_samples = k.less_equal(k.random_uniform([self.config.getint(
        #     'simulation', 'batch_size'), ]), 300 * self.dt / 1000.)
        # spiking_neurons = k.repeat(spiking_samples, 10)
        # activ = k.softmax(mem)
        # max_activ = k.max(activ, axis=1, keepdims=True)
        # output_spikes = k.equal(activ, max_activ).astype(self._floatx)
        # output_spikes = tf.where(k.equal(spiking_neurons, 0),
        #                          k.zeros_like(output_spikes), output_spikes)
        # new_and_reset_mem = tf.where(spiking_neurons, k.zeros_like(mem),
        #                                mem)
        # self.add_update([(self.mem, new_and_reset_mem)])
        # return output_spikes

        output_spikes = tf.less_equal(tf.random.uniform(tf.shape(mem)),
                                      tf.nn.softmax(mem))
        return tf.cast(output_spikes, self._floatx) * self.v_thresh

    def quantized_activation(self, mem, m, f):
        """Activation with precision reduced to fixed point format Qm.f."""
        # Todo: Needs to be implemented somehow...
        return tf.cast(tf.greater_equal(mem, self.v_thresh), self._floatx) * \
            self.v_thresh

    def get_new_mem(self):
        """Add input to membrane potential."""

        # Destroy impulse if in refractory period
        masked_impulse = self.impulse if self.tau_refrac == 0 else \
            tf.where(tf.greater(self.refrac_until, self.time),
                     tf.zeros_like(self.impulse), self.impulse)

        # Add impulse
        if clamp_var:
            # Experimental: Clamp the membrane potential to zero until the
            # presynaptic neurons fire at their steady-state rates. This helps
            # avoid a transient response.
            new_mem = tf.cond(tf.less(tf.reduce_mean(self.var), 1e-4) +
                              tf.greater(self.time, self.duration / 2),
                              lambda: self.mem + masked_impulse,
                              lambda: self.mem)
        elif hasattr(self, 'clamp_idx'):
            # Set clamp-duration by a specific delay from layer to layer.
            new_mem = tf.cond(tf.less(self.time, self.clamp_idx),
                              lambda: self.mem,
                              lambda: self.mem + masked_impulse)
        elif v_clip:
            # Clip membrane potential to prevent too strong accumulation.
            new_mem = tf.clip_by_value(self.mem + masked_impulse, -3, 3)
        else:
            new_mem = self.mem + masked_impulse

        if self.config.getboolean('cell', 'leak'):
            # Todo: Implement more flexible version of leak!
            new_mem = tf.where(tf.greater(new_mem, 0), new_mem - 0.1 * self.dt,
                               new_mem)

        return new_mem

    def set_reset_mem(self, mem, spikes):
        """
        Reset membrane potential ``mem`` array where ``spikes`` array is
        nonzero.
        """

        if (hasattr(self, 'activation_str') and
                self.activation_str == 'softmax'):
            # Turn off reset (uncomment second line) to get a faster and better
            # top-1 error. The top-5 error is better when resetting:
            new = tf.where(tf.not_equal(spikes, 0), tf.zeros_like(mem), mem)
            # new = tf.identity(mem)
        elif self.config.get('cell', 'reset') == 'Reset by subtraction':
            if self.payloads:  # Experimental.
                new = tf.where(tf.not_equal(spikes, 0),
                               tf.zeros_like(mem), mem)
            else:
                new = tf.where(tf.greater(spikes, 0), mem - self.v_thresh, mem)
                new = tf.where(tf.less(spikes, 0), new + self.v_thresh, new)
        elif self.config.get('cell', 'reset') == 'Reset by modulo':
            new = tf.where(tf.not_equal(spikes, 0), mem % self.v_thresh, mem)
        else:  # self.config.get('cell', 'reset') == 'Reset to zero':
            new = tf.where(tf.not_equal(spikes, 0), tf.zeros_like(mem), mem)
        self.mem.assign(new)

    def get_new_thresh(self):
        """Get new threshhold."""

        thr_min = self._v_thresh / 100
        thr_max = self._v_thresh
        r_lim = 1 / self.dt
        return thr_min + (thr_max - thr_min) * self.max_spikerate / r_lim

        # return tf.cond(
        #     k.equal(self.time / self.dt % settings['timestep_fraction'], 0) *
        #     k.greater(self.max_spikerate, settings['diff_to_min_rate']/1000)*
        #     k.greater(1 / self.dt - self.max_spikerate,
        #          settings['diff_to_max_rate'] / 1000),
        #     lambda: self.max_spikerate, lambda: self.v_thresh)

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
            if hasattr(self, 'b'):
                b = self.get_weights()[1]
                for i in range(len(b)):
                    init_mem[:, i, Ellipsis] = -b[i]
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
        if self.tau_refrac > 0:
            self.refrac_until.assign(tf.zeros(self.output_shape, self._floatx))
        if self.spiketrain is not None:
            self.spiketrain.assign(tf.zeros(self.output_shape, self._floatx))
        if self.payloads:
            self.payloads.assign(tf.zeros(self.output_shape, self._floatx))
            self.payloads_sum.assign(tf.zeros(self.output_shape, self._floatx))
        if self.online_normalization and do_reset:
            self.spikecounts.assign(tf.zeros(self.output_shape, self._floatx))
            self.max_spikerate.assign(0)
            self.v_thresh.assign(self._v_thresh)
        if clamp_var and do_reset:
            self.spikerate.assign(tf.zeros(self.input_shape, self._floatx))
            self.var.assign(tf.zeros(self.input_shape, self._floatx))

    @tf.function
    def init_neurons(self, input_shape):
        """Init layer neurons."""

        from snntoolbox.bin.utils import get_log_keys, get_plot_keys

        output_shape = self.compute_output_shape(input_shape)
        if self.v_thresh is None:  # Need this check because of @tf.function.
            self.v_thresh = tf.Variable(self._v_thresh, name='v_thresh',
                                        trainable=False)
        if self.mem is None:
            self.mem = tf.Variable(self.init_membrane_potential(output_shape),
                                   name='v_mem', trainable=False)
        if self.time is None:
            self.time = tf.Variable(self.dt, name='dt', trainable=False)
        # To save memory and computations, allocate only where needed:
        if self.tau_refrac > 0 and self.tau_refrac_until is None:
            self.refrac_until = tf.Variable(
                tf.zeros(output_shape), name='refrac_until', trainable=False)
        if any({'spiketrains', 'spikerates', 'correlation', 'spikecounts',
                'hist_spikerates_activations', 'operations',
                'synaptic_operations_b_t', 'neuron_operations_b_t',
                'spiketrains_n_b_l_t'} & (get_plot_keys(self.config) |
               get_log_keys(self.config))) and self.spiketrain is None:
            self.spiketrain = tf.Variable(tf.zeros(output_shape),
                                          trainable=False, name='spiketrains')
        if self.online_normalization and self.spikecounts is None:
            self.spikecounts = tf.Variable(tf.zeros(output_shape),
                                           trainable=False, name='spikecounts')
            self.max_spikerate = tf.Variable(tf.zeros([1]), trainable=False,
                                             name='max_spikerate')
        if self.config.getboolean('cell', 'payloads') \
                and self.payloads is None:
            self.payloads = tf.Variable(tf.zeros(output_shape),
                                        trainable=False, name='payloads')
            self.payloads_sum = tf.Variable(
                tf.zeros(output_shape), trainable=False, name='payloads_sum')
        if clamp_var and self.spikerate is None:
            self.spikerate = tf.Variable(tf.zeros(input_shape),
                                         trainable=False, name='spikerates')
            self.var = tf.Variable(tf.zeros(input_shape),
                                   trainable=False, name='var')
        if hasattr(self, 'clamp_idx'):
            self.clamp_idx = self.get_clamp_idx()

    def get_layer_idx(self):
        """Get index of layer."""

        label = self.name.split('_')[0]
        layer_idx = None
        for i in range(len(label)):
            if label[:i].isdigit():
                layer_idx = int(label[:i])
        return layer_idx

    def get_clamp_idx(self):
        """Get time step when to stop clamping membrane potential.

        Returns
        -------

        : int
            Time step when to stop clamping.
        """

        with open(self.filename_clamp_indices) as f:
            clamp_indices = json.load(f)

        clamp_idx = clamp_indices.get(str(self.get_layer_idx()))
        print("Clamping membrane potential until time step {}.".format(
            clamp_idx))

        return clamp_idx

    def update_avg_variance(self, spikes):
        """Keep a running average of the spike-rates and the their variance.

        Parameters
        ----------

        spikes:
            Output spikes.
        """

        delta = spikes - self.spikerate
        spikerate_new = self.spikerate + delta / self.time
        var_new = self.var + delta * (spikes - spikerate_new)
        self.var.assign(var_new / self.time)
        self.spikerate.assign(spikerate_new)

    @tf.function
    def update_b(self):
        """
        Get a new value for the bias, relaxing it over time to the true value.
        """

        i = self.get_layer_idx()
        m = tf.clip_by_value(1 - (1 - 2 * self.time / self.duration) * i / 50,
                             0, 1)
        self.bias.assign(self.bias * m)


def add_payloads(prev_layer, input_spikes):
    """Get payloads from previous layer."""

    # Get only payloads of those pre-synaptic neurons that spiked
    payloads = tf.where(tf.equal(input_spikes, 0.),
                        tf.zeros_like(input_spikes), prev_layer.payloads)
    print("Using spikes with payloads from layer {}".format(prev_layer.name))
    return input_spikes + payloads


def spike_call(call):
    @tf.function
    def decorator(self, x):

        if clamp_var:
            # Clamp membrane potential if spike rate variance too high
            self.update_avg_variance(x)
        if self.online_normalization:
            # Modify threshold if firing rate of layer too low
            self.v_thresh.assign(self.get_new_thresh())
        if self.payloads:
            # Add payload from previous layer
            x = add_payloads(get_inbound_layers(self)[0], x)

        self.impulse = call(self, x)
        return self.update_neurons()

    return decorator


def get_isi_from_impulse(impulse, epsilon):
    return tf.where(tf.less(impulse, epsilon), tf.zeros_like(impulse),
                    tf.divide(1., impulse))


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

    def call(self, x, mask=None):

        return super(SpikeFlatten, self).call(x)

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
    """Spike ZeroPadding2D layer."""

    def __init__(self, **kwargs):
        kwargs.pop(str('config'))
        ZeroPadding2D.__init__(self, **kwargs)

    def call(self, x, mask=None):

        return ZeroPadding2D.call(self, x)

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
    """Spike Reshape layer."""

    def __init__(self, **kwargs):
        kwargs.pop(str('config'))
        Reshape.__init__(self, **kwargs)

    def call(self, x, mask=None):

        return Reshape.call(self, x)

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
        self.init_neurons(input_shape.as_list())

        if self.config.getboolean('cell', 'bias_relaxation'):
            self.update_b()

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

        if self.config.getboolean('cell', 'bias_relaxation'):
            self.update_b()

    @spike_call
    def call(self, x, mask=None):

        return Conv2D.call(self, x)


class SpikeDepthwiseConv2D(DepthwiseConv2D, SpikeLayer):
    """Spike 2D DepthwiseConvolution."""

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

        if self.config.getboolean('cell', 'bias_relaxation'):
            self.update_b()

    @spike_call
    def call(self, x, mask=None):

        return DepthwiseConv2D.call(self, x)


class SpikeAveragePooling2D(AveragePooling2D, SpikeLayer):
    """Spike Average Pooling."""

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
        self.init_neurons(input_shape.as_list())

    @spike_call
    def call(self, x, mask=None):
        """Layer functionality."""

        print("WARNING: Rate-based spiking MaxPooling layer is not "
              "implemented in TensorFlow backend. Falling back on "
              "AveragePooling. Switch to Theano backend to use MaxPooling.")
        return tf.nn.avg_pool2d(x, self.pool_size, self.strides, self.padding)


custom_layers = {'SpikeFlatten': SpikeFlatten,
                 'SpikeReshape': SpikeReshape,
                 'SpikeZeroPadding2D': SpikeZeroPadding2D,
                 'SpikeDense': SpikeDense,
                 'SpikeConv2D': SpikeConv2D,
                 'SpikeDepthwiseConv2D': SpikeDepthwiseConv2D,
                 'SpikeAveragePooling2D': SpikeAveragePooling2D,
                 'SpikeMaxPooling2D': SpikeMaxPooling2D,
                 'SpikeConcatenate': SpikeConcatenate}
