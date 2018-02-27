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

from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import numpy as np
from future import standard_library
from keras import backend as k
from keras.layers import Dense, Flatten, AveragePooling2D, MaxPooling2D, Conv2D
from keras.layers import Layer, Concatenate

from snntoolbox.parsing.utils import get_inbound_layers

standard_library.install_aliases()

# Experimental
bias_relaxation = False
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
        if bias_relaxation:
            self.b0 = None
        if clamp_var:
            self.spikerate = self.var = None

        import os
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

        self.payloads = self.config.getboolean('cell', 'payloads')
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

    def reset(self, sample_idx):
        """Reset layer variables."""

        self.reset_spikevars(sample_idx)

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
            new_refractory = k.tf.where(k.not_equal(output_spikes, 0),
                                        self.time + self.tau_refrac,
                                        self.refrac_until)
            self.add_update([(self.refrac_until, new_refractory)])

        if self.payloads:
            residuals = k.tf.where(k.not_equal(output_spikes, 0),
                                   new_mem - self._v_thresh, new_mem)
            payloads, payloads_sum = self.update_payload(residuals,
                                                         output_spikes)
            self.add_update([(self.payloads, payloads),
                             (self.payloads_sum, payloads_sum)])

        if self.online_normalization:
            self.add_update([(self.spikecounts, self.spikecounts + k.cast(
                k.not_equal(output_spikes, 0), k.floatx())),
                             (self.max_spikerate,
                              k.max(self.spikecounts) * self.dt / self.time)])

        if self.spiketrain is not None:
            self.add_update([(self.spiketrain, self.time * k.cast(
                k.not_equal(output_spikes, 0), k.floatx()))])

        return k.cast(output_spikes, k.floatx())

    def update_payload(self, residuals, spikes):
        """Update payloads.

        Uses the residual of the membrane potential after spike.
        """

        idxs = k.not_equal(spikes, 0)
        payloads = k.tf.where(idxs, residuals[idxs] - self.payloads_sum[idxs],
                              self.payloads)
        payloads_sum = k.tf.where(idxs, self.payloads_sum + self.payloads,
                                  self.payloads_sum)
        return payloads, payloads_sum

    def linear_activation(self, mem):
        """Linear activation."""
        return k.cast(k.greater_equal(mem, self.v_thresh), k.floatx()) * \
            self.v_thresh

    def binary_sigmoid_activation(self, mem):
        """Binary sigmoid activation."""

        return k.cast(k.greater(mem, 0), k.floatx()) * self.v_thresh

    def binary_tanh_activation(self, mem):
        """Binary tanh activation."""

        output_spikes = k.cast(k.greater(mem, 0), k.floatx()) * self.v_thresh
        output_spikes += k.cast(k.less(mem, 0), k.floatx()) * -self.v_thresh

        return output_spikes

    def softmax_activation(self, mem):
        """Softmax activation."""

        # spiking_samples = k.less_equal(k.random_uniform([self.config.getint(
        #     'simulation', 'batch_size'), ]), 300 * self.dt / 1000.)
        # spiking_neurons = k.repeat(spiking_samples, 10)
        # activ = k.softmax(mem)
        # max_activ = k.max(activ, axis=1, keepdims=True)
        # output_spikes = k.equal(activ, max_activ).astype(k.floatx())
        # output_spikes = k.tf.where(k.equal(spiking_neurons, 0),
        #                          k.zeros_like(output_spikes), output_spikes)
        # new_and_reset_mem = k.tf.where(spiking_neurons, k.zeros_like(mem),
        #                                mem)
        # self.add_update([(self.mem, new_and_reset_mem)])
        # return output_spikes

        return k.cast(k.less_equal(k.random_uniform(k.shape(mem)),
                                   k.softmax(mem)), k.floatx()) * self.v_thresh

    def quantized_activation(self, mem, m, f):
        """Activation with precision reduced to fixed point format Qm.f."""
        # Todo: Needs to be implemented somehow...
        return k.cast(k.greater_equal(mem, self.v_thresh), k.floatx()) * \
            self.v_thresh

    def get_new_mem(self):
        """Add input to membrane potential."""

        # Destroy impulse if in refractory period
        masked_impulse = self.impulse if self.tau_refrac == 0 else \
            k.tf.where(k.greater(self.refrac_until, self.time),
                       k.zeros_like(self.impulse), self.impulse)

        # Add impulse
        if clamp_var:
            # Experimental: Clamp the membrane potential to zero until the
            # presynaptic neurons fire at their steady-state rates. This helps
            # avoid a transient response.
            new_mem = k.tf.cond(k.less(k.mean(self.var), 1e-4) +
                                k.greater(self.time, self.duration / 2),
                                lambda: self.mem + masked_impulse,
                                lambda: self.mem)
        elif hasattr(self, 'clamp_idx'):
            # Set clamp-duration by a specific delay from layer to layer.
            new_mem = k.tf.cond(k.less(self.time, self.clamp_idx),
                                lambda: self.mem,
                                lambda: self.mem + masked_impulse)
        elif v_clip:
            # Clip membrane potential to prevent too strong accumulation.
            new_mem = k.clip(self.mem + masked_impulse, -3, 3)
        else:
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
            # Turn off reset (uncomment second line) to get a faster and better
            # top-1 error. The top-5 error is better when resetting:
            new = k.tf.where(k.not_equal(spikes, 0), k.zeros_like(mem), mem)
            # new = k.tf.identity(mem)
        elif self.config.get('cell', 'reset') == 'Reset by subtraction':
            if self.payloads:  # Experimental.
                new = k.tf.where(k.not_equal(spikes, 0), k.zeros_like(mem), mem)
            else:
                new = k.tf.where(k.greater(spikes, 0), mem - self.v_thresh, mem)
                new = k.tf.where(k.less(spikes, 0), new + self.v_thresh, new)
        elif self.config.get('cell', 'reset') == 'Reset by modulo':
            new = k.tf.where(k.not_equal(spikes, 0), mem % self.v_thresh, mem)
        else:  # self.config.get('cell', 'reset') == 'Reset to zero':
            new = k.tf.where(k.not_equal(spikes, 0), k.zeros_like(mem), mem)
        self.add_update([(self.mem, new)])

    def get_new_thresh(self):
        """Get new threshhold."""

        thr_min = self._v_thresh / 100
        thr_max = self._v_thresh
        r_lim = 1 / self.dt
        return thr_min + (thr_max - thr_min) * self.max_spikerate / r_lim

        # return k.tf.cond(
        #     k.equal(self.time / self.dt % settings['timestep_fraction'], 0) *
        #     k.greater(self.max_spikerate, settings['diff_to_min_rate']/1000) *
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
            if hasattr(self, 'b'):
                b = self.get_weights()[1]
                for i in range(len(b)):
                    init_mem[:, i, Ellipsis] = -b[i]
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
        if self.tau_refrac > 0:
            k.set_value(self.refrac_until,
                        np.zeros(self.output_shape, k.floatx()))
        if self.spiketrain is not None:
            k.set_value(self.spiketrain,
                        np.zeros(self.output_shape, k.floatx()))
        if self.payloads:
            k.set_value(self.payloads,
                        np.zeros(self.output_shape, k.floatx()))
            k.set_value(self.payloads_sum,
                        np.zeros(self.output_shape, k.floatx()))
        if self.online_normalization and do_reset:
            k.set_value(self.spikecounts,
                        np.zeros(self.output_shape, k.floatx()))
            k.set_value(self.max_spikerate, np.float32(0.))
            k.set_value(self.v_thresh, np.float32(self._v_thresh))
        if clamp_var and do_reset:
            k.set_value(self.spikerate, np.zeros(self.input_shape, k.floatx()))
            k.set_value(self.var, np.zeros(self.input_shape, k.floatx()))

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
        if self.online_normalization:
            self.spikecounts = k.zeros(output_shape)
            self.max_spikerate = k.variable(0)
        if self.payloads:
            self.payloads = k.zeros(output_shape)
            self.payloads_sum = k.zeros(output_shape)
        if clamp_var:
            self.spikerate = k.zeros(input_shape)
            self.var = k.zeros(input_shape)
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

        import json

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
        self.add_update([(self.var, var_new / self.time),
                         (self.spikerate, spikerate_new)])

    def update_b(self):
        """
        Get a new value for the bias, relaxing it over time to the true value.
        """

        i = self.get_layer_idx()
        return self.b0 * k.minimum(k.maximum(
            0, 1 - (1 - 2 * self.time / self.duration) * i / 50), 1)


def add_payloads(prev_layer, input_spikes):
    """Get payloads from previous layer."""

    # Get only payloads of those pre-synaptic neurons that spiked
    payloads = k.tf.where(k.equal(input_spikes, 0.), k.zeros_like(input_spikes),
                          prev_layer.payloads)
    print("Using spikes with payloads from layer {}".format(prev_layer.name))
    return input_spikes + payloads


def spike_call(call):
    def decorator(self, x):

        if clamp_var:
            # Clamp membrane potential if spike rate variance too high
            self.update_avg_variance(x)
        if self.online_normalization:
            # Modify threshold if firing rate of layer too low
            self.add_update([(self.v_thresh, self.get_new_thresh())])
        if self.payloads:
            # Add payload from previous layer
            x = add_payloads(get_inbound_layers(self)[0], x)

        self.impulse = call(self, x)
        return self.update_neurons()

    return decorator


def get_isi_from_impulse(impulse, epsilon):
    return k.tf.where(k.less(impulse, epsilon), k.zeros_like(impulse),
                      np.true_divide(1., impulse))


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

        if bias_relaxation:
            self.b0 = k.variable(k.get_value(self.bias))
            self.add_update([(self.bias, self.update_b())])

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

        if bias_relaxation:
            self.b0 = k.variable(k.get_value(self.bias))
            self.add_update([(self.bias, self.update_b())])

    @spike_call
    def call(self, x, mask=None):

        return Conv2D.call(self, x)


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
        self.init_neurons(input_shape)

    @spike_call
    def call(self, x, mask=None):

        return AveragePooling2D.call(self, x)


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

    @spike_call
    def call(self, x, mask=None):
        """Layer functionality."""

        print("WARNING: Rate-based spiking MaxPooling layer is not implemented "
              "in TensorFlow backend. Falling back on AveragePooling. Switch "
              "to Theano backend to use MaxPooling.")
        return k.pool2d(x, self.pool_size, self.strides, self.padding,
                        pool_mode='avg')


custom_layers = {'SpikeFlatten': SpikeFlatten,
                 'SpikeDense': SpikeDense,
                 'SpikeConv2D': SpikeConv2D,
                 'SpikeAveragePooling2D': SpikeAveragePooling2D,
                 'SpikeMaxPooling2D': SpikeMaxPooling2D,
                 'SpikeConcatenate': SpikeConcatenate}
