# -*- coding: utf-8 -*-
"""INI spiking neuron simulator.

A collection of helper functions, including spiking layer classes derived from
Keras layers, which were used to implement our own IF spiking simulator.

Not needed when converting and running the SNN in other simulators (pyNN,
MegaSim, ...)

Created on Tue Dec  8 10:41:10 2015

@author: rbodo
"""

# For compatibility with python2
from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import os
import warnings
import numpy as np
from future import standard_library
import theano
import theano.tensor as t
from theano.tensor.signal import pool
from theano.tensor.shared_randomstreams import RandomStreams
from keras import backend as k
from keras.layers import Concatenate
from keras.layers import Dense, Flatten, AveragePooling2D, MaxPooling2D, Conv2D
from snntoolbox.config import settings

standard_library.install_aliases()

rng = RandomStreams()

floatX = theano.config.floatX

# Experimental
bias_relaxation = False
clamp_var = False
v_clip = False


def update_neurons(self):
    """Update neurons according to activation function."""

    if hasattr(self, 'activation_str'):
        if self.activation_str == 'softmax':
            output_spikes = softmax_activation(self)
        elif self.activation_str == 'binary_sigmoid':
            output_spikes = binary_sigmoid_activation(self)
        elif self.activation_str == 'binary_tanh':
            output_spikes = binary_tanh_activation(self)
        else:
            output_spikes = linear_activation(self)
    else:
        output_spikes = linear_activation(self)

    # Store refractory
    if settings['tau_refrac'] > 0:
        new_refractory = t.set_subtensor(
            self.refrac_until[output_spikes.nonzero()],
            self.time + self.tau_refrac)
        add_updates(self, [(self.refrac_until, new_refractory)])

    if settings['online_normalization']:
        add_updates(self, [(self.spikecounts,
                            t.add(self.spikecounts, t.neq(output_spikes, 0)))])
        add_updates(self, [(self.max_spikerate, t.max(self.spikecounts) *
                            settings['dt'] / self.time)])

    if self.spiketrain is not None:
        add_updates(self, [(self.spiketrain,
                            self.time * t.neq(output_spikes, 0))])

    return t.cast(output_spikes, floatX)


def update_payload(self, residuals, idxs):
    """Update payloads.

    Uses the residual of the membrane potential after spike.
    """

    payloads = t.set_subtensor(
        self.payloads[idxs], residuals[idxs] - self.payloads_sum[idxs])
    payloads_sum = t.set_subtensor(
        self.payloads_sum[idxs], self.payloads_sum[idxs] + self.payloads[idxs])
    return payloads, payloads_sum


def linear_activation(self):
    """Linear activation."""

    new_mem = get_new_mem(self)

    # Store spiking
    output_spikes = t.mul(t.ge(new_mem, self.v_thresh), self.v_thresh)

    set_reset_mem(self, new_mem, output_spikes)

    if settings['payloads']:
        spike_idxs = output_spikes.nonzero()
        residuals = t.inc_subtensor(new_mem[spike_idxs], -self.v_thresh)
        payloads, payloads_sum = update_payload(self, residuals, spike_idxs)
        add_updates(self, [(self.payloads, payloads)])
        add_updates(self, [(self.payloads_sum, payloads_sum)])

    return output_spikes


def binary_sigmoid_activation(self):
    """Binary sigmoid activation."""

    new_mem = get_new_mem(self)

    output_spikes = t.mul(t.gt(new_mem, 0), self.v_thresh)

    set_reset_mem(self, new_mem, output_spikes)

    return output_spikes


def binary_tanh_activation(self):
    """Binary tanh activation."""

    new_mem = get_new_mem(self)

    output_spikes = t.mul(t.gt(new_mem, 0), self.v_thresh)
    output_spikes += t.mul(t.lt(new_mem, 0), -self.v_thresh)

    set_reset_mem(self, new_mem, output_spikes)

    return output_spikes


def softmax_activation(self):
    """Softmax activation."""

    new_mem = get_new_mem(self)

    output_spikes = t.mul(t.le(rng.uniform(new_mem.shape),
                               t.nnet.softmax(new_mem)), self.v_thresh)

    set_reset_mem(self, new_mem, output_spikes)

    return output_spikes


def get_new_mem(self):
    """Add input to membrane potential."""

    # Destroy impulse if in refractory period
    masked_impulse = self.impulse if settings['tau_refrac'] == 0 else \
        t.set_subtensor(self.impulse[t.nonzero(self.refrac_until > self.time)],
                        0.)

    # Add impulse
    if clamp_var:
        # Experimental: Clamp the membrane potential to zero until the
        # presynaptic neurons fire at their steady-state rates. This helps avoid
        # a transient response.
        new_mem = theano.ifelse.ifelse(
            t.lt(t.mean(self.var), 1e-4) +
            t.gt(self.time, settings['duration'] / 2),
            self.mem + masked_impulse, self.mem)
    elif settings['filename_clamp_indices'] != '':
        # Set clamp-duration by a specific delay from layer to layer.
        new_mem = theano.ifelse.ifelse(t.lt(self.time, self.clamp_idx),
                                       self.mem, self.mem + masked_impulse)
    elif v_clip:
        # Clip membrane potential to [-2, 2] to prevent too strong accumulation.
        new_mem = theano.tensor.clip(self.mem + masked_impulse, -3, 3)
    else:
        new_mem = self.mem + masked_impulse

    return new_mem


def set_reset_mem(self, mem, spikes):
    """Reset membrane potential ``mem`` array where ``spikes`` array is nonzero.
    """

    spike_idxs = t.nonzero(spikes)
    if settings['reset'] == 'Reset by subtraction':
        if settings['payloads'] and False:  # Experimental, turn off by default
            new = t.set_subtensor(mem[spike_idxs], 0.)
        else:
            pos_spike_idxs = t.nonzero(t.gt(spikes, 0))
            neg_spike_idxs = t.nonzero(t.lt(spikes, 0))
            new = t.inc_subtensor(mem[pos_spike_idxs], -self.v_thresh)
            new = t.inc_subtensor(new[neg_spike_idxs], self.v_thresh)
    elif settings['reset'] == 'Reset by modulo':
        new = t.set_subtensor(mem[spike_idxs], mem[spike_idxs] % self.v_thresh)
    else:  # settings['reset'] == 'Reset to zero':
        new = t.set_subtensor(mem[spike_idxs], 0.)
    add_updates(self, [(self.mem, new)])


def get_new_thresh(self):
    """Get new threshhold."""

    thr_min = 0.5
    thr_max = 1.0
    r_lim = 1 / settings['dt']
    return thr_min + (thr_max - thr_min) * self.max_spikerate / r_lim

    # return theano.ifelse.ifelse(
    #     t.eq(self.time / settings['dt'] % settings['timestep_fraction'], 0) *
    #     t.gt(self.max_spikerate, settings['diff_to_min_rate'] / 1000) *
    #     t.gt(1 / settings['dt'] - self.max_spikerate,
    #          settings['diff_to_max_rate'] / 1000),
    #     self.max_spikerate, self.v_thresh)


def get_time(self):
    """Get simulation time variable.

    Parameters
    ----------

    self: SpikeLayer
        SpikeLayer derived from keras.layers.Layer.

    Returns
    -------

    : Union[None, float]
        If layer has ``time`` attribute, return current simulation time, else
        ``None``.
    """

    return self.time.get_value() if hasattr(self, 'time') else None


def set_time(self, time):
    """Set simulation time variable.

    Parameters
    ----------

    self: SpikeLayer
        SpikeLayer derived from keras.layers.Layer.
    time: float
        Current simulation time.
    """

    self.time.set_value(time)


def add_payloads(prev_layer, input_spikes):
    """Get payloads from previous layer."""

    # Get only payloads of those pre-synaptic neurons that spiked
    payloads = t.set_subtensor(
        prev_layer.payloads[t.nonzero(t.eq(input_spikes, 0.))], 0.)
    print("Using spikes with payloads from layer {}".format(prev_layer.name))
    return t.add(input_spikes, payloads)


def add_updates(self, updates):
    """Update self.updates.
    This is taken from a development-version of Keras. Might be able to remove
    it with the next official version. (27.11.16)"""

    if not hasattr(self, 'updates'):
        self.updates = []
    try:
        self.updates += updates
    except AttributeError:
        pass


def init_membrane_potential(self, mode='zero'):
    """Initialize membrane potential.

    Helpful to avoid transient response in the beginning of the simulation.
    Not needed when reset between frames is turned off, e.g. with a video data
    set.

    Parameters
    ----------

    self: Subclass[keras.layers.core.Layer]
        The layer.
    mode: str
        Initialization mode.

        - ``'uniform'``: Random numbers from uniform distribution in
          ``[-thr, thr]``.
        - ``'bias'``: Negative bias.
        - ``'zero'``: Zero (default).

    Returns
    -------

    init_mem: theano.tensor.sharedvar
        A tensor of ``self.output_shape`` (same as layer).
    """

    if mode == 'uniform':
        init_mem = k.random_uniform(self.output_shape,
                                    -self.v_thresh, self.v_thresh)
    elif mode == 'bias':
        init_mem = np.zeros(self.output_shape, floatX)
        if hasattr(self, 'b'):
            b = self.get_weights()[1]
            for i in range(len(b)):
                init_mem[:, i, Ellipsis] = np.float32(-b[i])
    else:  # mode == 'zero':
        init_mem = np.zeros(self.output_shape, floatX)
    return init_mem


def reset_spikevars(self, sample_idx):
    """
    Reset variables present in spiking layers. Can be turned off for instance 
    when a video sequence is tested."""

    mod = settings['reset_between_nth_sample']
    mod = mod if mod else sample_idx + 1
    do_reset = sample_idx % mod == 0
    if do_reset:
        self.mem.set_value(init_membrane_potential(self))
    self.time.set_value(np.float32(settings['dt']))
    if settings['tau_refrac'] > 0:
        self.refrac_until.set_value(np.zeros(self.output_shape, floatX))
    if self.spiketrain is not None:
        self.spiketrain.set_value(np.zeros(self.output_shape, floatX))
    if settings['payloads']:
        self.payloads.set_value(np.zeros(self.output_shape, floatX))
        self.payloads_sum.set_value(np.zeros(self.output_shape, floatX))
    if settings['online_normalization'] and do_reset:
        self.spikecounts.set_value(np.zeros(self.output_shape, floatX))
        self.max_spikerate.set_value(np.float32(0.))
        self.v_thresh.set_value(np.float32(settings['v_thresh']))
    if clamp_var and do_reset:
        self.spikerate.set_value(np.zeros(self.input_shape, floatX))
        self.var.set_value(np.zeros(self.input_shape, floatX))


def init_neurons(self, input_shape, tau_refrac=0.):
    """Init layer neurons."""

    output_shape = self.compute_output_shape(input_shape)
    self.v_thresh = theano.shared(np.float32(settings['v_thresh']))
    self.tau_refrac = tau_refrac
    self.mem = k.zeros(output_shape)
    self.time = theano.shared(np.float32(settings['dt']))
    # To save memory and computations, allocate only where needed:
    if settings['tau_refrac'] > 0:
        self.refrac_until = k.zeros(output_shape)
    if any({'spiketrains', 'spikerates', 'correlation', 'spikecounts',
            'hist_spikerates_activations', 'operations', 'operations_b_t',
            'spiketrains_n_b_l_t'}
            & (settings['plot_vars'] | settings['log_vars'])):
        self.spiketrain = k.zeros(output_shape)
    if settings['online_normalization']:
        self.spikecounts = k.zeros(output_shape)
        self.max_spikerate = theano.shared(np.float32(0))
    if settings['payloads']:
        self.payloads = k.zeros(output_shape)
        self.payloads_sum = k.zeros(output_shape)
    if clamp_var:
        self.spikerate = k.zeros(input_shape)
        self.var = k.zeros(input_shape)
    if settings['filename_clamp_indices'] != '':
        self.clamp_idx = get_clamp_idx(self)


def get_layer_idx(self):
    """Get index of layer."""

    l = self.name.split('_')
    layer_idx = None
    for i in range(max(4, len(l) - 2)):
        if l[0][:i].isnumeric():
            layer_idx = int(l[0][:i])
    return layer_idx


def get_clamp_idx(self):
    """Get time step when to stop clamping membrane potential.

    Parameters
    ----------

    self:
        Layer

    Returns
    -------

    : int
        Time step when to stop clamping.
    """

    clamp_idx = np.loadtxt(os.path.join(
        settings['path_wd'], settings['filename_clamp_indices']), 'int')
    layer_idx = get_layer_idx(self)
    return clamp_idx[layer_idx]


def update_avg_variance(self, spikes):
    """Keep a running average of the spike-rates and the their variance.

    Parameters
    ----------

    self:
        Layer
    spikes:
        Output spikes.
    """

    delta = spikes - self.spikerate
    spikerate_new = self.spikerate + delta / self.time
    var_new = self.var + delta * (spikes - spikerate_new)
    add_updates(self, [(self.var, var_new / self.time)])
    add_updates(self, [(self.spikerate, spikerate_new)])


def update_b(self):
    """Get a new value for the bias, relaxing it over time to the true value."""
    i = get_layer_idx(self)
    return self.b0 * k.minimum(k.maximum(
        0, 1 - (1 - 2 * self.time / settings['duration']) * i / 50), 1)


class SpikeConcatenate(Concatenate):
    """Spike merge layer"""

    def _merge_function(self, inputs):
        return self._merge_function(inputs)

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

    def call(self, x, mask=None):
        """Layer functionality."""

        return t.cast(super(SpikeFlatten, self).call(x), floatX)

    @staticmethod
    def reset(sample_idx):
        """Reset layer variables."""

        pass

    @property
    def class_name(self):
        """Get class name."""

        return self.__class__.__name__


class SpikeDense(Dense):
    """Spike Dense layer."""

    def __init__(self, units, **kwargs):
        """Init function."""
        # Replace activation from kwargs by 'linear' before initializing
        # superclass, because the relu activation is applied by the spike-
        # generation mechanism automatically. In some cases (binary activation),
        # we need to apply a the activation manually. This information is taken
        # from the 'activation' key during conversion.
        self.activation_str = str(kwargs.pop('activation'))
        super(SpikeDense, self).__init__(units, **kwargs)
        self.layer_type = self.class_name
        self.tau_refrac = kwargs['tau_refrac'] if 'tau_refrac' in kwargs else 0.
        self.v_thresh = None
        self.stateful = True
        self._per_input_updates = {}
        self.time = None
        self.mem = self.spiketrain = self.impulse = self.spikecounts = None
        self.refrac_until = self.max_spikerate = None
        if bias_relaxation:
            self.b0 = None
        if clamp_var:
            self.spikerate = self.var = None
        if settings['filename_clamp_indices'] != '':
            self.clamp_idx = None

    def build(self, input_shape):
        """Creates the layer weights.
        Must be implemented on all layers that have weights.

        Parameters
        ----------

        input_shape: Union[list, tuple, Any]
            Keras tensor (future input to layer) or list/tuple of Keras tensors
            to reference for weight shape computations.
        """

        super(SpikeDense, self).build(input_shape)
        init_neurons(self, input_shape)
        if bias_relaxation:
            self.b0 = k.variable(self.bias.get_value())
            add_updates(self, [(self.bias, update_b(self))])

    def call(self, x, mask=None):
        """Layer functionality."""

        if clamp_var:
            update_avg_variance(self, x)

        inp = x

        if settings['online_normalization']:
            # Modify threshold if firing rate of layer too low
            add_updates(self, [(self.v_thresh, get_new_thresh(self))])
        if settings['payloads']:
            # Add payload from previous layer
            inp = add_payloads(self.inbound_nodes[0].inbound_layers[0], inp)

        self.impulse = super(SpikeDense, self).call(inp)
        return update_neurons(self)

    def reset(self, sample_idx):
        """Reset layer variables."""

        reset_spikevars(self, sample_idx)

    @property
    def class_name(self):
        """Get class name."""

        return self.__class__.__name__


class SpikeConv2D(Conv2D):
    """Spike 2D Convolution."""

    def __init__(self, filters, kernel_size, filter_flip=True, **kwargs):
        """Init function."""
        # Replace activation from kwargs by 'linear' before initializing
        # superclass, because the relu activation is applied by the spike-
        # generation mechanism automatically. In some cases (binary activation),
        # we need to apply a the activation manually. This information is taken
        # from the 'activation' key during conversion.
        self.activation_str = str(kwargs.pop('activation'))
        super(SpikeConv2D, self).__init__(filters, kernel_size, **kwargs)
        self.layer_type = self.class_name
        self.filter_flip = filter_flip
        self.tau_refrac = kwargs['tau_refrac'] if 'tau_refrac' in kwargs else 0.
        self.v_thresh = None
        self.stateful = True
        self._per_input_updates = {}
        self.time = None
        self.mem = self.spiketrain = self.impulse = self.spikecounts = None
        self.refrac_until = self.max_spikerate = None
        if bias_relaxation:
            self.b0 = None
        if clamp_var:
            self.spikerate = self.var = None
        if settings['filename_clamp_indices'] != '':
            self.clamp_idx = None

    def build(self, input_shape):
        """Creates the layer weights.
        Must be implemented on all layers that have weights.

        Parameters
        ----------

        input_shape: Union[list, tuple, Any]
            Keras tensor (future input to layer) or list/tuple of Keras tensors
            to reference for weight shape computations.
        """

        super(SpikeConv2D, self).build(input_shape)
        init_neurons(self, input_shape)
        if bias_relaxation:
            self.b0 = k.variable(self.bias.get_value())
            add_updates(self, [(self.bias, update_b(self))])

    def call(self, x, mask=None):
        """Layer functionality."""

        if clamp_var:
            update_avg_variance(self, x)

        inp = x

        if settings['payloads']:
            # Add payload from previous layer
            inp = add_payloads(self.inbound_nodes[0].inbound_layers[0], inp)

        if settings['online_normalization']:
            # Modify threshold if firing rate of layer too low
            add_updates(self, [(self.v_thresh, get_new_thresh(self))])

        self.impulse = super(SpikeConv2D, self).call(inp)
        return update_neurons(self)

    def reset(self, sample_idx):
        """Reset layer variables."""

        reset_spikevars(self, sample_idx)

    @property
    def class_name(self):
        """Get class name."""

        return self.__class__.__name__


class SpikeAveragePooling2D(AveragePooling2D):
    """Average Pooling."""

    def __init__(self, **kwargs):
        """Init average pooling."""

        super(SpikeAveragePooling2D, self).__init__(**kwargs)
        self.layer_type = self.class_name
        self.tau_refrac = kwargs['tau_refrac'] if 'tau_refrac' in kwargs else 0.
        self.v_thresh = None
        self.stateful = True
        self._per_input_updates = {}
        self.time = None
        self.mem = self.spiketrain = self.impulse = self.spikecounts = None
        self.refrac_until = self.max_spikerate = None
        if clamp_var:
            self.spikerate = self.var = None
        if settings['filename_clamp_indices'] != '':
            self.clamp_idx = None

    def build(self, input_shape):
        """Creates the layer weights.
        Must be implemented on all layers that have weights.

        Parameters
        ----------

        input_shape: Union[list, tuple, Any]
            Keras tensor (future input to layer) or list/tuple of Keras tensors
            to reference for weight shape computations.
        """

        super(SpikeAveragePooling2D, self).build(input_shape)
        init_neurons(self, input_shape)

    def call(self, x, mask=None):
        """Layer functionality."""

        if clamp_var:
            update_avg_variance(self, x)

        inp = x

        if settings['payloads']:
            # Add payload from previous layer
            inp = add_payloads(self.inbound_nodes[0].inbound_layers[0], inp)

        self.impulse = super(SpikeAveragePooling2D, self).call(inp)
        return update_neurons(self)

    def reset(self, sample_idx):
        """Reset layer variables."""

        reset_spikevars(self, sample_idx)

    @property
    def class_name(self):
        """Get class name."""

        return self.__class__.__name__


class SpikeMaxPooling2D(MaxPooling2D):
    """Max Pooling."""

    def __init__(self, **kwargs):
        """Init function."""

        super(SpikeMaxPooling2D, self).__init__(**kwargs)
        self.layer_type = self.class_name
        self.ignore_border = True if self.padding == 'valid' else False
        if settings['custom_activation'] is not None \
                and 'binary' in settings['custom_activation']:
            self.activation_str = settings['custom_activation']
        self.tau_refrac = kwargs['tau_refrac'] if 'tau_refrac' in kwargs else 0.
        self.v_thresh = None
        self.stateful = True
        self._per_input_updates = {}
        self.spikerate_pre = self.time = self.previous_x = None
        self.mem = self.spiketrain = self.impulse = self.spikecounts = None
        self.refrac_until = self.max_spikerate = None
        if clamp_var:
            self.spikerate = self.var = None
        if settings['filename_clamp_indices'] != '':
            self.clamp_idx = None

    def build(self, input_shape):
        """Creates the layer weights.
        Must be implemented on all layers that have weights.

        Parameters
        ----------

        input_shape: Union[list, tuple, Any]
            Keras tensor (future input to layer) or list/tuple of Keras tensors
            to reference for weight shape computations.
        """

        super(SpikeMaxPooling2D, self).build(input_shape)
        init_neurons(self, input_shape)
        self.spikerate_pre = theano.shared(np.zeros(input_shape, floatX))
        self.previous_x = theano.shared(np.zeros(input_shape, floatX))

    def call(self, x, mask=None):
        """Layer functionality."""

        if clamp_var:
            update_avg_variance(self, x)

        inp = x

        if settings['payloads']:
            # Add payload from previous layer
            inp = add_payloads(self.inbound_nodes[0].inbound_layers[0], inp)

        if settings['custom_activation'] is not None \
                and 'binary' in settings['custom_activation']:
            self.impulse = k.pool2d(inp, self.pool_size, self.strides,
                                    self.padding, pool_mode='max')
        elif settings['maxpool_type'] in ['avg_max', 'fir_max', 'exp_max']:
            if settings['maxpool_type'] == 'avg_max':
                update_rule = self.spikerate_pre + \
                              (x - self.spikerate_pre) * \
                              settings['dt'] / self.time
            elif settings['maxpool_type'] == 'exp_max':
                # update_rule = self.spikerate_pre + x / 2. ** (1 / t_inv)
                update_rule = self.spikerate_pre * 1.005 + x * 0.995
            else:  # settings['maxpool_type'] == 'fir_max':
                update_rule = self.spikerate_pre + \
                              x * settings['dt'] / self.time
            add_updates(self, [(self.spikerate_pre, update_rule)])
            add_updates(self, [(self.previous_x, x)])
            self.impulse = self._pooling_function(
                [self.spikerate_pre, self.previous_x], self.pool_size,
                self.strides, self.padding, self.data_format)
        else:
            print("Wrong max pooling type, "
                  "falling back on Average Pooling instead.")
            self.impulse = k.pool2d(inp, self.pool_size, self.strides,
                                    self.padding, pool_mode='avg')
        return update_neurons(self)

    def _pooling_function(self, inputs, pool_size, strides, padding,
                          data_format):
        return spike_pool2d(inputs, pool_size, strides, padding, data_format,
                            'max')

    def reset(self, sample_idx):
        """Reset layer variables."""

        reset_spikevars(self, sample_idx)
        mod = settings['reset_between_nth_sample']
        mod = mod if mod else sample_idx + 1
        if sample_idx % mod == 0:
            self.spikerate_pre.set_value(np.zeros(self.input_shape, floatX))

    @property
    def class_name(self):
        """Get class name."""

        return self.__class__.__name__


def spike_pool2d(inputs, pool_size, strides=(1, 1), padding='valid',
                 data_format=None, pool_mode='max'):
    """MaxPooling with spikes.

    Parameters
    ----------

    inputs :
    pool_size :
    strides :
    padding :
    data_format :
    pool_mode :

    Returns
    -------

    """

    if data_format is None:
        data_format = k.image_data_format()
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Unknown data_format:', data_format)

    assert pool_size[0] >= 1 and pool_size[1] >= 1

    x = inputs[0]  # Presynaptic spike-rates
    y = inputs[1]  # Presynaptic spikes

    if padding == 'same':
        w_pad = pool_size[0] - 2 if pool_size[0] > 2 and pool_size[0] % 2 == 1 \
            else pool_size[0] - 1
        h_pad = pool_size[1] - 2 if pool_size[1] > 2 and pool_size[1] % 2 == 1 \
            else pool_size[1] - 1
        pad = (w_pad, h_pad)
    elif padding == 'valid':
        pad = (0, 0)
    else:
        raise ValueError('Invalid border mode: ', padding)

    if data_format == 'channels_last':
        x = x.dimshuffle((0, 3, 1, 2))
        y = y.dimshuffle((0, 3, 1, 2))

    if pool_mode == 'max':
        pool_out = spike_pool_2d(inputs, pool_size, True, strides, pad, 'max')
    elif pool_mode == 'avg':
        pool_out = pool.pool_2d(y, ws=pool_size, stride=strides, pad=pad,
                                ignore_border=True, mode='average_exc_pad')
    else:
        raise Exception('Invalid pooling mode: ' + str(pool_mode))

    if padding == 'same':
        expected_width = (x.shape[2] + strides[0] - 1) // strides[0]
        expected_height = (x.shape[3] + strides[1] - 1) // strides[1]
        pool_out = pool_out[:, :, : expected_width, : expected_height]

    if data_format == 'channels_last':
        pool_out = pool_out.dimshuffle((0, 2, 3, 1))

    return pool_out


def spike_pool_2d(inputs, ws, ignore_border=None, stride=None, pad=(0, 0),
                  mode='max'):
    """Downscale the input by a specified factor

    Takes as input a N-D tensor, where N >= 2. It downscales the input image by
    the specified factor, by keeping only the maximum value of non-overlapping
    patches of size (ds[0],ds[1])

    Parameters
    ----------
    inputs : list[N-D theano tensors of input images]
        Input images. Max pooling will be done over the 2 last dimensions.
    ws : tuple of length 2
        Factor by which to downscale (vertical ds, horizontal ds).
        (2,2) will halve the image in each dimension.
    ignore_border : bool (default None, will print a warning and set to False)
        When True, (5,5) input with ds=(2,2) will generate a (2,2) output.
        (3,3) otherwise.
    stride : tuple of two ints
        Stride size, which is the number of shifts over rows/cols to get the
        next pool region. If st is None, it is considered equal to ds
        (no overlap on pooling regions).
    pad : tuple of two ints
        (pad_h, pad_w), pad zeros to extend beyond four borders of the
        images, pad_h is the size of the top and bottom margins, and
        pad_w is the size of the left and right margins.
    mode : {'max', 'sum', 'average_inc_pad', 'average_exc_pad'}
        Operation executed on each window. `max` and `sum` always exclude
        the padding in the computation. `average` gives you the choice to
        include or exclude it.

    """

    x = inputs[0]  # Presynaptic spike-rates

    if x.ndim < 2:
        raise NotImplementedError('pool_2d requires a dimension >= 2')
    if ignore_border is None:
        warnings.warn(
            "pool_2d() will have the parameter ignore_border"
            " default value changed to True (currently"
            " False). To have consistent behavior with all Theano"
            " version, explicitly add the parameter ignore_border=True."
            " On the GPU, using ignore_border=True is needed to use cuDNN."
            " When using ignore_border=False and not using cuDNN, the only"
            " GPU combination supported is when"
            " `ws == stride and pad == (0, 0) and mode == 'max'`."
            " Otherwise, the convolution will be executed on CPU.",
            stacklevel=2)
        ignore_border = False
    op = SpikePool(ws, ignore_border, stride, pad, mode, 2)
    return op(inputs, ws, stride, pad)


class SpikePool(theano.Op):
    """
    For N-dimensional tensors, consider that the last two dimensions span
    images. This Op downsamples these images by taking the max, sum or average
    over different patch.

    The constructor takes the max, sum or average or different input patches.

    Parameters
    ----------
    ws : list or tuple of two ints
        Downsample factor over rows and column.
        ds indicates the pool region size.
    ignore_border : bool
        If ds doesn't divide imgshape, do we include an extra row/col
        of partial downsampling (False) or ignore it (True).
    stride : list or tuple of two ints or None
        Stride size, which is the number of shifts over rows/cols to get the
        next pool region. If st is None, it is considered equal to ds
        (no overlap on pooling regions).
    pad: tuple of two ints
        (pad_h, pad_w), pad zeros to extend beyond four borders of the images,
        pad_h is the size of the top and bottom margins, and pad_w is the size
        of the left and right margins.
    mode : {'max', 'sum', 'average_inc_pad', 'average_exc_pad'}
        ('average_inc_pad' excludes the padding from the count,
        'average_exc_pad' include it)

    """

    __props__ = ('ws', 'ignore_border', 'stride', 'pad', 'mode', 'ndim')

    def __init__(self, ws, ignore_border=False, stride=None, pad=(0, 0),
                 mode='max', ndim=2):
        super(SpikePool, self).__init__()
        self.ws = tuple(ws)
        if stride is None:
            stride = ws
        assert isinstance(stride, (tuple, list))
        self.stride = tuple(stride)
        self.ignore_border = ignore_border
        self.pad = tuple(pad)
        self.ndim = ndim
        if self.pad != (0, 0) and not ignore_border:
            raise NotImplementedError(
                'padding works only with ignore_border=True')
        if self.pad[0] >= self.ws[0] or self.pad[1] >= self.ws[1]:
            raise NotImplementedError(
                'padding_h and padding_w must be smaller than strides')
        if mode not in ['max', 'average_inc_pad', 'average_exc_pad', 'sum']:
            raise ValueError(
                "Pool mode parameter only support 'max', 'sum',"
                " 'average_inc_pad' and 'average_exc_pad'. Got %s" % mode)
        self.mode = mode

    def R_op(self, inputs, eval_points):
        """

        Parameters
        ----------

        inputs :
        eval_points :
        """

        super(SpikePool, self).R_op(inputs, eval_points)

    def make_node(self, x, ws, stride=None, pad=None):
        """

        Parameters
        ----------
        pad : 
        stride : 
        ws : 
        x :

        Returns
        -------

        """

        for i in range(len(x)):
            x[i] = t.as_tensor_variable(x[i])

        nd = self.ndim
        if stride is None:
            stride = ws
        if pad is None:
            pad = (0,) * nd
        elif isinstance(pad, (tuple, list)):
            if max(pad) != 0 and not self.ignore_border:
                raise NotImplementedError(
                    'padding works only with ignore_border=True')
            if isinstance(ws, (tuple, list)):
                if any(pad[i] >= ws[i] for i in range(nd)):
                    raise NotImplementedError(
                        'padding must be smaller than strides')
        ws = t.as_tensor_variable(ws)
        stride = t.as_tensor_variable(stride)
        pad = t.as_tensor_variable(pad)
        assert ws.ndim == 1
        assert stride.ndim == 1
        assert pad.ndim == 1
        if x[0].type.ndim < nd:
            raise TypeError()
        if ws.dtype not in t.int_dtypes:
            raise TypeError('Pool downsample parameters must be ints.')
        if stride.dtype not in t.int_dtypes:
            raise TypeError('Stride parameters must be ints.')
        if pad.dtype not in t.int_dtypes:
            raise TypeError('Padding parameters must be ints.')
        # If the input shape are broadcastable we can have 0 in the output shape
        broad = x[0].broadcastable[:-nd] + (False,) * nd
        out = t.TensorType(x[0].dtype, broad)
        return theano.gof.Apply(self, x+[ws, stride, pad], [out()])

    def perform(self, node, inp, out, **kwargs):
        """Perform pooling operation on spikes.

        Parameters
        ----------

        **kwargs :
        out :
        inp :
        node :
        """

        # xr contains the presynaptic spike-rates, and xs the presynaptic spikes
        xr, xs, ws, stride, pad = inp
        z, = out
        nd = self.ndim
        assert ws.shape == stride.shape == pad.shape == (nd,)
        if len(xr.shape) < nd:
            raise NotImplementedError(
                'Pool requires input with {} or more dimensions'.format(nd))
        z_shape = pool.Pool.out_shape(xr.shape, ws, self.ignore_border, stride,
                                      pad, nd)
        if not self.ignore_border:
            assert all(z > 0 for z in z_shape[-nd:])
        if (z[0] is None) or (z[0].shape != z_shape):
            z[0] = np.zeros(z_shape, dtype=xr.dtype)
        zz = z[0]
        # size of pooling output
        pool_out_shp = zz.shape[-nd:]
        img_shp = tuple(xr.shape[-nd + i] + 2 * pad[i] for i in range(nd))
        inc_pad = self.mode == 'average_inc_pad'

        # pad the image
        if max(self.pad) != 0:
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
                if not inc_pad:
                    start = max(start, pad[i])
                    end = min(end, img_shp[i] - pad[i])
                region_slices[i].append(slice(start, end))

        spike = settings['v_thresh']
        # iterate over non-pooling dimensions
        for n in np.ndindex(*xr.shape[:-nd]):
            yrn = yr[n]
            ysn = ys[n]
            zzn = zz[n]
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
                    zzn[r] = spike


custom_layers = {'SpikeFlatten': SpikeFlatten,
                 'SpikeDense': SpikeDense,
                 'SpikeConv2D': SpikeConv2D,
                 'SpikeAveragePooling2D': SpikeAveragePooling2D,
                 'SpikeMaxPooling2D': SpikeMaxPooling2D,
                 'SpikeConcatenate': SpikeConcatenate}
