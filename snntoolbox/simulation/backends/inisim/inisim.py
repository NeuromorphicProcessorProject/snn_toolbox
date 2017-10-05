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
        if self.config.getboolean('conversion', 'use_isi_code'):
            self.last_spiketimes = None
            self.thresh_b_l = None
            self.sum_of_abs_weights = None
            self.prev_impulse = None

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

        psp = self.get_psp(output_spikes)

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

        # if self.config.getboolean('conversion', 'use_isi_code'):
        #     self.add_update([(self.v_thresh, self.v_thresh - np.true_divide(
        #         k.abs(self.prev_impulse - self.impulse), self.dt)),
        #                      (self.prev_impulse, self.impulse)])

        if self.spiketrain is not None:
            self.add_update([(self.spiketrain, self.time * k.cast(
                k.not_equal(output_spikes, 0), k.floatx()))])

        return k.cast(psp, k.floatx())

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
            new_mem = k.ifelse.ifelse(k.less(k.mean(self.var), 1e-4) +
                                      k.greater(self.time, self.duration / 2),
                                      self.mem + masked_impulse, self.mem)
        elif hasattr(self, 'clamp_idx'):
            # Set clamp-duration by a specific delay from layer to layer.
            new_mem = k.ifelse.ifelse(k.less(self.time, self.clamp_idx),
                                      self.mem, self.mem + masked_impulse)
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
            # new = mem.copy()
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

        # return k.ifelse.ifelse(
        #     k.equal(self.time / self.dt % settings['timestep_fraction'], 0) *
        #     k.greater(self.max_spikerate, settings['diff_to_min_rate']/1000) *
        #     k.greater(1 / self.dt - self.max_spikerate,
        #          settings['diff_to_max_rate'] / 1000),
        #     self.max_spikerate, self.v_thresh)

    def get_psp(self, output_spikes):
        if self.config.getboolean('conversion', 'use_isi_code'):
            new_spiketimes = k.tf.where(
                k.not_equal(output_spikes, 0),
                k.ones_like(output_spikes) * self.get_time(),
                self.last_spiketimes)
            self.add_update([(self.last_spiketimes, new_spiketimes)])
            # psp = k.maximum(0, np.true_divide(self.dt, self.last_spiketimes))
            psp = k.tf.where(k.greater(self.last_spiketimes, 0),
                             k.ones_like(output_spikes) * self.dt,
                             k.zeros_like(output_spikes))
            return psp
        else:
            return output_spikes

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
        if self.config.getboolean('conversion', 'use_isi_code'):
            k.set_value(self.last_spiketimes, -np.ones(self.output_shape,
                                                       k.floatx()))
            # k.set_value(self.v_thresh, self._v_thresh * np.ones(
            #     self.output_shape, k.floatx()) + self.sum_of_abs_weights)
            # k.set_value(self.prev_impulse, np.zeros(self.output_shape,
            #                                      k.floatx()))

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
        if self.config.getboolean('conversion', 'use_isi_code'):
            self.last_spiketimes = k.variable(-np.ones(output_shape))
            # self.sum_of_abs_weights = 0 if len(self.weights) == 0 else \
            #     np.sum(np.abs(self.get_weights()[0]))
            # self.v_thresh = k.variable(
            #     self._v_thresh * np.ones(output_shape) +
            #     self.sum_of_abs_weights)
            # self.prev_impulse = k.zeros(output_shape)

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

        if self.config.getboolean('conversion', 'temporal_pattern_coding'):
            # Transform x into binary format here. Effective batch_size
            # increases from 1 to 32.
            x = to_binary(x)

            # Multiply binary feature map matrix by PSP kernel which decays
            # exponentially across the 32 temporal steps (batch-dimension).
            num_bits = len(x)
            x *= k.reshape([2**-i for i in range(num_bits)],
                           (num_bits, 1, 1, 1))
            self.impulse = call(self, x)
            return k.sum(self.impulse, 0)
            # Need to apply activation function here.
            # Get rid of membrane state variables.
            # May need to increase batch size to 32 already when building the
            # net, or call "call" 32 times. Though in Pooling layers, need to
            # sum over first axis before applying call.
            # For measuring spike trains, store the binary matrix. For rates,
            # the return value of "call", after summing over first axis and
            # applying activation. To count operations, Apply fan-out to binary
            # matrix and count each operation twice (MAC due to convolution of
            # kernel with non-unity PSPs).
            # For output, consider activation values instead of spike sums.
        else:
            self.impulse = call(self, x)
            return self.update_neurons()

    return decorator


def to_binary(x, num_bits=None):
    """Transform an array of floats into binary representation.

    Parameters
    ----------

    x: ndarray
        Input array containing float values. The first dimension has to be of
        length 1.
    num_bits: int
        The fixed point precision to be used when converting to binary. Will be
        inferred from ``x`` if not specified.

    Returns
    -------

    binary_array: ndarray
        Output boolean array. The first dimension of x is expanded to length
        ``bits``. The binary representation of each value in ``x`` is
        distributed across the first dimension of ``binary_array``.
    """

    import itertools

    # assert x.shape[0] == 1, "The first dimension of the input array has to " \
    #                         "be of length 1, found {}.".format(x.shape[0])

    if num_bits is None:
        num_bits = np.finfo(x.dtype).bits

    to_binary_str = np.vectorize(lambda z: binary_repr(
        k.cast(z * 2 ** (num_bits - 1), 'int32'), num_bits))
    binary_str = to_binary_str(x)
    binary_array = np.repeat(np.empty_like(x, bool), num_bits, 0)
    shape = k.get_variable_shape(binary_array)
    for i, j, l in itertools.product(
            range(shape[1]), range(shape[2]), range(shape[3])):
        binary_array[:, i, j, l] = np.array(list(binary_str[0, i, j, l]))  # Todo: Find more efficient way than this for loop

    return k.variable(binary_array)


def binary_repr(num, width=None):
    """
    Return the binary representation of the input number as a string.

    Parameters
    ----------

    num : int
        Only an integer decimal number can be used.
    width : int, optional
        The length of the returned string.

    Returns
    -------

    bin : str
        Binary representation of `num` or two's complement of `num`.
    """

    if k.equal(num, 0):
        return '0' * (width or 1)
    elif k.greater(num, 0):
        binary = bin(num)[2:]
        binwidth = len(binary)
        outwidth = (binwidth if width is None else max(binwidth, width))
        return binary.zfill(outwidth)


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
        return None

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
        return None

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

    def __init__(self, **kwargs):
        MaxPooling2D.__init__(self, **kwargs)
        self.spikerate_pre = self.previous_x = None
        self.activation_str = None

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
        self.spikerate_pre = k.zeros(input_shape)
        self.previous_x = k.zeros(input_shape)

    @spike_call
    def call(self, x, mask=None):
        """Layer functionality."""

        maxpool_type = self.config.get('conversion', 'maxpool_type')
        if 'binary' in self.activation_str or \
                self.config.getboolean('conversion', 'use_isi_code'):
            return k.pool2d(x, self.pool_size, self.strides, self.padding,
                            pool_mode='max')
        elif maxpool_type == 'avg_max':
            update_rule = self.spikerate_pre + (x - self.spikerate_pre) * \
                          self.dt / self.time
        elif maxpool_type == 'exp_max':
            # update_rule = self.spikerate_pre + x / 2. ** (1 / t_inv)
            update_rule = self.spikerate_pre * 1.005 + x * 0.995
        elif maxpool_type == 'fir_max':
            update_rule = self.spikerate_pre + x * self.dt / self.time
        else:
            print("Wrong max pooling type, falling back on average pooling.")
            return k.pool2d(x, self.pool_size, self.strides, self.padding,
                            pool_mode='avg')
        self.add_update([(self.spikerate_pre, update_rule),
                         (self.previous_x, x)])
        return self._pooling_function([self.spikerate_pre, self.previous_x],
                                      self.pool_size, self.strides,
                                      self.padding, self.data_format)

    def _pooling_function(self, inputs, pool_size, strides, padding,
                          data_format):
        return spike_pool2d(inputs, pool_size, strides, padding, data_format)

    def reset(self, sample_idx):
        """Reset layer variables."""

        self.reset_spikevars(sample_idx)
        mod = self.config.getint('simulation', 'reset_between_nth_sample')
        mod = mod if mod else sample_idx + 1
        if sample_idx % mod == 0:
            k.set_value(self.spikerate_pre,
                        np.zeros(self.input_shape, k.floatx()))

    @property
    def class_name(self):
        """Get class name."""

        return self.__class__.__name__


# noinspection PyProtectedMember
def spike_pool2d(inputs, pool_size, strides=(1, 1), padding='valid',
                 data_format=None):
    """2D max pooling with spikes.

    # Arguments
        inputs: Tensor or variable.
        pool_size: tuple of 2 integers.
        strides: tuple of 2 integers.
        padding: string, `"same"` or `"valid"`.
        data_format: string, `"channels_last"` or `"channels_first"`.

    # Returns
        A tensor, result of 2D pooling.

    # Raises
        ValueError: if `data_format` is neither `"channels_last"` or
        `"channels_first"`.
        ValueError: if `pool_mode` is neither `"max"` or `"avg"`.
    """

    if data_format is None:
        data_format = k.image_data_format()
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Unknown data_format ' + str(data_format))

    padding = k.tensorflow_backend._preprocess_padding(padding)
    strides = (1,) + strides + (1,)
    pool_size = (1,) + pool_size + (1,)

    x = inputs[0]  # Presynaptic spike-rates
    y = inputs[1]  # Presynaptic spikes

    x = k.tensorflow_backend._preprocess_conv2d_input(x, data_format)
    y = k.tensorflow_backend._preprocess_conv2d_input(y, data_format)

    x = spike_max_pool(x, y, pool_size, strides, padding)

    x = k.tensorflow_backend._postprocess_conv2d_output(x, data_format)

    return x


def spike_max_pool(value_x, value_y, ksize, strides, padding, name=None):
    """Performs the max pooling on the input.

    Args:
      value_x: A 4-D `Tensor` with shape `[batch, height, width, channels]` and
        type `tf.float32`.
      value_y: A 4-D `Tensor` with shape `[batch, height, width, channels]` and
        type `tf.float32`.
      ksize: A list of ints that has length >= 4.  The size of the window for
        each dimension of the input tensor.
      strides: A list of ints that has length >= 4.  The stride of the sliding
        window for each dimension of the input tensor.
      padding: A string, either `'VALID'` or `'SAME'`. The padding algorithm.
        See the @{tf.nn.convolution$comment here}
      name: Optional name for the operation.

    Returns:
      A `Tensor` with type `tf.float32`. The max pooled output tensor.
    """

    with k.tf.name_scope(name, "SpikeMaxPool", [value_x, value_y]) as name:
        value_x = k.tf.convert_to_tensor(value_x, name="value_x")
        value_y = k.tf.convert_to_tensor(value_y, name="value_y")
        return k.tf.py_func(_spike_max_pool, [value_x, value_y, ksize, strides,
                            padding], k.tf.float32, False, name)


def _spike_max_pool(xr, xs, ksize, strides, padding):
    """Performs max pooling on the input.

    Args:
    xr: A `Tensor`. Must be one of the following types: `float32`,
      `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`, `uint16`, `half`.
      4-D input to pool over.
    xs: A `Tensor`. Must be one of the following types: `float32`,
      `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`, `uint16`, `half`.
      4-D input to pool over.
    ksize: A list of `ints` that has length `>= 4`.
      The size of the window for each dimension of the input tensor.
    strides: A list of `ints` that has length `>= 4`.
      The stride of the sliding window for each dimension of the
      input tensor.
    padding: A `string` from: `"SAME", "VALID"`.
      The type of padding algorithm to use.

    Returns:
    A `Tensor`. Has the same type as `input`. The max pooled output tensor.
    """

    stride = strides[1:3]
    ws = ksize[1:3]

    padding = padding.decode('utf-8')

    if padding == 'SAME':
        w_pad = ws[0] - 2 if ws[0] > 2 and ws[0] % 2 == 1 else ws[0] - 1
        h_pad = ws[1] - 2 if ws[1] > 2 and ws[1] % 2 == 1 else ws[1] - 1
        pad = (w_pad, h_pad)
    elif padding == 'VALID':
        pad = (0, 0)
    else:
        raise ValueError('Invalid border mode: ', padding)

    # xr contains the presynaptic spike-rates, and xs the presynaptic spikes
    nd = 2
    if len(xr.shape) < nd:
        raise NotImplementedError(
            'Pool requires input with {} or more dimensions'.format(nd))

    z_shape = out_shape(xr.shape, ws, stride, pad, nd)

    z = np.zeros(z_shape, dtype=xr.dtype)
    # size of pooling output
    pool_out_shp = z.shape[-nd:]
    img_shp = tuple(xr.shape[-nd + i] + 2 * pad[i] for i in range(nd))

    # pad the image
    if max(pad) != 0:
        yr = np.zeros(xr.shape[:-nd] + img_shp, dtype=xr.dtype)
        # noinspection PyTypeChecker
        yr[(slice(None),)*(len(xr.shape)-nd) + tuple(
            slice(pad[i], img_shp[i]-pad[i]) for i in range(nd))] = xr
        ys = np.zeros(xs.shape[:-nd] + img_shp, dtype=xs.dtype)
        # noinspection PyTypeChecker
        ys[(slice(None),)*(len(xs.shape)-nd) + tuple(slice(
            pad[i], img_shp[i]-pad[i]) for i in range(nd))] = xs
    else:
        yr = xr
        ys = xs

    # precompute the region boundaries for each dimension
    region_slices = [[] for _ in range(nd)]
    for i in range(nd):
        for j in range(pool_out_shp[i]):
            start = j * stride[i]
            end = min(start + ws[i], img_shp[i])
            start = max(start, pad[i])
            end = min(end, img_shp[i] - pad[i])
            region_slices[i].append(slice(start, end))

    # TODO: Spike size should equal threshold, which may vary during simulation.
    spike = 1  # kwargs[str('v_thresh')]

    # iterate over non-pooling dimensions
    for n in np.ndindex(*xr.shape[:-nd]):
        yrn = yr[n]
        ysn = ys[n]
        zn = z[n]
        # iterate over pooling regions
        for r in np.ndindex(*pool_out_shp):
            rate_patch = yrn[[region_slices[i][r[i]] for i in range(nd)]]
            if not rate_patch.any():
                # Need to prevent the layer to output a spike at
                # index 0 if all rates are equally zero.
                continue
            spike_patch = ysn[[region_slices[i][r[i]] for i in range(nd)]]
            # The second condition is not completely equivalent to the first
            # because the former has a higher chance of admitting spikes.
            # if (spike_patch*(rate_patch == np.argmax(rate_patch))).any():
            if spike_patch.flatten()[np.argmax(rate_patch)]:
                zn[r] = spike

    if padding == 'SAME':
        expected_width = (xr.shape[1] + stride[0] - 1) // stride[0]
        expected_height = (xr.shape[2] + stride[1] - 1) // stride[1]
        z = z[:, : expected_width, : expected_height, :]

    return z


def out_shape(imgshape, ws, stride, pad, ndim=2):
    patch_shape = tuple(imgshape[-ndim + i] + pad[i] * 2
                        for i in range(ndim))
    outshape = [compute_out(patch_shape[i], ws[i], stride[i])
                for i in range(ndim)]
    return list(imgshape[:-ndim]) + outshape


def compute_out(v, downsample, stride):
    if downsample == stride:
        return v // stride
    else:
        out = (v - downsample) // stride + 1
        try:
            if k.is_keras_tensor(out):
                return k.maximum(out, 0)
        except ValueError:
            return np.maximum(out, 0)


custom_layers = {'SpikeFlatten': SpikeFlatten,
                 'SpikeDense': SpikeDense,
                 'SpikeConv2D': SpikeConv2D,
                 'SpikeAveragePooling2D': SpikeAveragePooling2D,
                 'SpikeMaxPooling2D': SpikeMaxPooling2D,
                 'SpikeConcatenate': SpikeConcatenate}
