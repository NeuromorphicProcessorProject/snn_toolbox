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

import warnings
import numpy as np
import theano
import theano.tensor as t
from future import standard_library
from keras import backend as k
from keras.layers import Convolution2D, Merge
from keras.layers import Dense, Flatten, AveragePooling2D, MaxPooling2D
from snntoolbox.config import settings
from theano.tensor.shared_randomstreams import RandomStreams
from theano.tensor.signal import pool

standard_library.install_aliases()

rng = RandomStreams()

floatX = theano.config.floatX

bias_relaxation = False
clamp_var = False
clamp_delay = False
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
                            t.add(self.spikecounts, output_spikes))])
        add_updates(self, [(self.max_spikerate, t.max(self.spikecounts) *
                            settings['dt'] / self.time)])

    if self.spiketrain is not None:
        add_updates(self, [(self.spiketrain, self.time * output_spikes)])

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


def binary_sigmoid_activation(self):
    """Binary sigmoid activation."""

    # Destroy impulse if in refractory period
    masked_imp = self.impulse if settings['tau_refrac'] == 0 else \
        t.set_subtensor(self.impulse[t.nonzero(self.refrac_until > self.time)],
                        0.)

    # Add impulse
    new_mem = self.mem + masked_imp

    # Store spiking
    output_spikes = t.gt(new_mem, 0)

    spike_idxs = output_spikes.nonzero()

    # Reset neurons
    new_and_reset_mem = t.set_subtensor(new_mem[spike_idxs], 0.)

    add_updates(self, [(self.mem, new_and_reset_mem)])

    return output_spikes


def binary_tanh_activation(self):
    """Binary tanh activation."""

    # Destroy impulse if in refractory period
    masked_imp = self.impulse if settings['tau_refrac'] == 0 else \
        t.set_subtensor(self.impulse[t.nonzero(self.refrac_until > self.time)],
                        0.)

    # Add impulse
    new_mem = self.mem + masked_imp

    # Store spiking
    signed_spikes = t.set_subtensor(
        new_mem[t.nonzero(t.gt(new_mem, 0))], self.v_thresh)
    signed_spikes = t.set_subtensor(
        signed_spikes[t.nonzero(t.lt(signed_spikes, 0))], -self.v_thresh)
    output_spikes = t.set_subtensor(new_mem[t.nonzero(new_mem)], self.v_thresh)

    # Reset neurons
    new_and_reset_mem = t.set_subtensor(new_mem[output_spikes.nonzero()], 0.)

    add_updates(self, [(self.mem, new_and_reset_mem)])

    return signed_spikes


def linear_activation(self):
    """Linear activation."""

    # Destroy impulse if in refractory period
    masked_imp = self.impulse if settings['tau_refrac'] == 0 else \
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
            self.mem + masked_imp, self.mem)
    elif clamp_delay:
        # Set clamp-duration by a specific delay from layer to layer.
        new_mem = theano.ifelse.ifelse(t.lt(self.time, self.clamp_idx),
                                       self.mem, self.mem + masked_imp)
    elif v_clip:
        # Clip membrane potential to [-2, 2] to prevent too strong accumulation.
        new_mem = theano.tensor.clip(self.mem + masked_imp, -2, 2)
    else:
        new_mem = self.mem + masked_imp

    # Store spiking
    output_spikes = t.ge(new_mem, self.v_thresh)
    spike_idxs = output_spikes.nonzero()

    if settings['reset'] == 'Reset by subtraction':
        if settings['payloads'] and False:  # Experimental, turn off by default
            new_and_reset_mem = t.set_subtensor(new_mem[spike_idxs], 0.)
        else:
            new_and_reset_mem = t.inc_subtensor(new_mem[spike_idxs],
                                                -self.v_thresh)
    elif settings['reset'] == 'Reset by modulo':
        new_and_reset_mem = t.set_subtensor(new_mem[spike_idxs],
                                            new_mem[spike_idxs] % self.v_thresh)
    else:  # settings['reset'] == 'Reset to zero':
        new_and_reset_mem = t.set_subtensor(new_mem[spike_idxs], 0.)

    add_updates(self, [(self.mem, new_and_reset_mem)])

    if settings['payloads']:
        residuals = t.inc_subtensor(new_mem[spike_idxs], -self.v_thresh)
        payloads, payloads_sum = update_payload(self, residuals, spike_idxs)
        add_updates(self, [(self.payloads, payloads)])
        add_updates(self, [(self.payloads_sum, payloads_sum)])

    return output_spikes


def softmax_activation(self):
    """Softmax activation."""

    # Destroy impulse if in refractory period
    masked_imp = self.impulse if settings['tau_refrac'] == 0 else \
        t.set_subtensor(self.impulse[t.nonzero(self.refrac_until > self.time)],
                        0.)

    # Add impulse
    new_mem = self.mem + masked_imp

    # Store spiking
    spiking_samples = t.le(rng.uniform([settings['batch_size'], 1]),
                           settings['softmax_clockrate'] * settings[
                               'dt'] / 1000.)
    spiking_neurons = t.repeat(spiking_samples, 10, axis=1)
    activ = t.nnet.softmax(new_mem)
    max_activ = t.max(activ, axis=1, keepdims=True)
    output_spikes = t.eq(activ, max_activ).astype(floatX)
    output_spikes = t.set_subtensor(
        output_spikes[t.eq(spiking_neurons, 0).nonzero()], 0.)
    new_and_reset_mem = t.set_subtensor(new_mem[spiking_neurons.nonzero()], 0.)
    add_updates(self, [(self.mem, new_and_reset_mem)])

    return output_spikes


def get_new_thresh(self):
    """Get new threshhold."""

    return theano.ifelse.ifelse(
        t.eq(self.time / settings['dt'] % settings['timestep_fraction'], 0) *
        t.gt(self.max_spikerate, settings['diff_to_min_rate'] / 1000) *
        t.gt(1 / settings['dt'] - self.max_spikerate,
             settings['diff_to_max_rate'] / 1000),
        self.max_spikerate, self.v_thresh)


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


def reset_spikevars(self):
    """Reset variables present in spiking layers."""
    if settings['reset_between_frames']:
        self.mem.set_value(init_membrane_potential(self))
    self.time.set_value(np.float32(settings['dt']))
    if settings['tau_refrac'] > 0:
        self.refrac_until.set_value(np.zeros(self.output_shape, floatX))
    if self.spiketrain is not None:
        self.spiketrain.set_value(np.zeros(self.output_shape, floatX))
    if settings['online_normalization']:
        self.spikecounts.set_value(np.zeros(self.output_shape, floatX))
    if settings['payloads']:
        self.payloads.set_value(np.zeros(self.output_shape, floatX))
        self.payloads_sum.set_value(np.zeros(self.output_shape, floatX))
    if settings['online_normalization']:
        self.max_spikerate.set_value(0.)
        self.v_thresh.set_value(settings['v_thresh'])
    if clamp_var:
        self.spikerate.set_value(np.zeros(self.input_shape, floatX))
        self.var.set_value(np.zeros(self.input_shape, floatX))


def init_neurons(self, input_shape, tau_refrac=0.):
    """Init layer neurons."""

    output_shape = self.get_output_shape_for(input_shape)
    self.v_thresh = theano.shared(settings['v_thresh'])
    self.tau_refrac = tau_refrac
    self.mem = k.zeros(output_shape)
    self.time = theano.shared(np.float32(settings['dt']))
    # To save memory and computations, allocate only where needed:
    if settings['tau_refrac'] > 0:
        self.refrac_until = k.zeros(output_shape)
    if any({'spiketrains', 'spikerates', 'correlation',
            'hist_spikerates_activations'} & settings['plot_vars']) \
            or 'spiketrains_n_b_l_t' in settings['log_vars']:
        self.spiketrain = k.zeros(output_shape)
    if settings['online_normalization']:
        self.spikecounts = k.zeros(output_shape)
    if settings['payloads']:
        self.payloads = k.zeros(output_shape)
        self.payloads_sum = k.zeros(output_shape)
    if settings['online_normalization']:
        self.max_spikerate = k.zeros(1)
    if clamp_var:
        self.spikerate = k.zeros(input_shape)
        self.var = k.zeros(input_shape)
    if clamp_delay:
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

    clamp_idx = np.loadtxt(settings['path_wd'] + '/clamp_idx.txt', 'int')
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


class SpikeMerge(Merge):
    """Spike merge layer"""

    @staticmethod
    def reset():
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
    def reset():
        """Reset layer variables."""

        pass

    @property
    def class_name(self):
        """Get class name."""

        return self.__class__.__name__


class SpikeDense(Dense):
    """Spike Dense layer."""

    def __init__(self, output_dim, **kwargs):
        """Init function."""
        # Replace activation from kwargs by 'linear' before initializing
        # superclass, because the relu activation is applied by the spike-
        # generation mechanism automatically. In some cases (binary activation),
        # we need to apply a the activation manually. This information is taken
        # from the 'activation' key during conversion.
        self.activation_str = str(kwargs.pop('activation'))
        super(SpikeDense, self).__init__(output_dim, **kwargs)
        self.layer_type = self.class_name
        self.tau_refrac = kwargs['tau_refrac'] if 'tau_refrac' in kwargs else 0.
        self.v_thresh = None
        self.stateful = True
        self.updates = []
        self._per_input_updates = {}
        self.time = None
        self.mem = self.spiketrain = self.impulse = self.spikecounts = None
        self.refrac_until = self.max_spikerate = None
        if bias_relaxation:
            self.b0 = None
        if clamp_var:
            self.spikerate = self.var = None
        if clamp_delay:
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
            self.b0 = k.variable(self.b.get_value())
            add_updates(self, [(self.b, update_b(self))])

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

    def reset(self):
        """Reset layer variables."""

        reset_spikevars(self)

    @property
    def class_name(self):
        """Get class name."""

        return self.__class__.__name__


class SpikeConvolution2D(Convolution2D):
    """Spike 2D Convolution."""

    def __init__(self, nb_filter, nb_row, nb_col, filter_flip=True, **kwargs):
        """Init function."""
        # Replace activation from kwargs by 'linear' before initializing
        # superclass, because the relu activation is applied by the spike-
        # generation mechanism automatically. In some cases (binary activation),
        # we need to apply a the activation manually. This information is taken
        # from the 'activation' key during conversion.
        self.activation_str = str(kwargs.pop('activation'))
        super(SpikeConvolution2D, self).__init__(nb_filter, nb_row, nb_col,
                                                 **kwargs)
        self.layer_type = self.class_name
        self.filter_flip = filter_flip
        self.tau_refrac = kwargs['tau_refrac'] if 'tau_refrac' in kwargs else 0.
        self.v_thresh = None
        self.stateful = True
        self.updates = []
        self._per_input_updates = {}
        self.time = None
        self.mem = self.spiketrain = self.impulse = self.spikecounts = None
        self.refrac_until = self.max_spikerate = None
        if bias_relaxation:
            self.b0 = None
        if clamp_var:
            self.spikerate = self.var = None
        if clamp_delay:
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

        super(SpikeConvolution2D, self).build(input_shape)
        init_neurons(self, input_shape)
        if bias_relaxation:
            self.b0 = k.variable(self.b.get_value())
            add_updates(self, [(self.b, update_b(self))])

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

        self.impulse = super(SpikeConvolution2D, self).call(inp)
        return update_neurons(self)

    def reset(self):
        """Reset layer variables."""

        reset_spikevars(self)

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
        self.updates = []
        self._per_input_updates = {}
        self.time = None
        self.mem = self.spiketrain = self.impulse = self.spikecounts = None
        self.refrac_until = self.max_spikerate = None
        if clamp_var:
            self.spikerate = self.var = None
        if clamp_delay:
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

    def reset(self):
        """Reset layer variables."""

        reset_spikevars(self)

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
        self.ignore_border = True if self.border_mode == 'valid' else False
        if 'binary' in settings['maxpool_type']:
            self.activation_str = settings['maxpool_type']
        self.tau_refrac = kwargs['tau_refrac'] if 'tau_refrac' in kwargs else 0.
        self.v_thresh = None
        self.stateful = True
        self.updates = []
        self._per_input_updates = {}
        self.spikerate_pre = self.time = self.previous_x = None
        self.mem = self.spiketrain = self.impulse = self.spikecounts = None
        self.refrac_until = self.max_spikerate = None
        if clamp_var:
            self.spikerate = self.var = None
        if clamp_delay:
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

        if 'binary' in settings['maxpool_type']:
            self.impulse = super(SpikeMaxPooling2D, self).call(inp)
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
                self.strides, self.border_mode, self.dim_ordering)
        else:
            print("Wrong max pooling type, "
                  "falling back on Average Pooling instead.")
            self.impulse = k.pool2d(inp, self.pool_size, self.strides,
                                    self.border_mode, pool_mode='avg')
        return update_neurons(self)

    def _pooling_function(self, inputs, pool_size, strides, border_mode,
                          dim_ordering):
        return spike_pool2d(inputs, pool_size, strides, border_mode,
                            dim_ordering, 'max')

    def reset(self):
        """Reset layer variables."""

        reset_spikevars(self)
        self.spikerate_pre.set_value(np.zeros(self.input_shape, floatX))

    @property
    def class_name(self):
        """Get class name."""

        return self.__class__.__name__


def spike_pool2d(inputs, pool_size, strides=(1, 1), border_mode='valid',
                 dim_ordering=k.image_dim_ordering(), pool_mode='max'):
    """MaxPooling with spikes.

    Parameters
    ----------

    inputs :
    pool_size :
    strides :
    border_mode :
    dim_ordering :
    pool_mode :

    Returns
    -------

    """

    x = inputs[0]  # Presynaptic spike-rates
    y = inputs[1]  # Presynaptic spikes

    if border_mode == 'same':
        w_pad = pool_size[0] - 2 if pool_size[0] % 2 == 1 else pool_size[0] - 1
        h_pad = pool_size[1] - 2 if pool_size[1] % 2 == 1 else pool_size[1] - 1
        padding = (w_pad, h_pad)
    elif border_mode == 'valid':
        padding = (0, 0)
    else:
        raise Exception('Invalid border mode: ' + str(border_mode))

    if dim_ordering not in {'th', 'tf'}:
        raise Exception('Unknown dim_ordering ' + str(dim_ordering))

    if dim_ordering == 'tf':
        x = x.dimshuffle((0, 3, 1, 2))
        y = y.dimshuffle((0, 3, 1, 2))

    if pool_mode == 'max':
        pool_out = spike_pool_2d(inputs, pool_size, True, strides, padding,
                                 'max')
    elif pool_mode == 'avg':
        pool_out = pool.pool_2d(y, ds=pool_size, st=strides,
                                ignore_border=True,
                                padding=padding,
                                mode='average_exc_pad')
    else:
        raise Exception('Invalid pooling mode: ' + str(pool_mode))

    if border_mode == 'same':
        expected_width = (x.shape[2] + strides[0] - 1) // strides[0]
        expected_height = (x.shape[3] + strides[1] - 1) // strides[1]

        pool_out = pool_out[:, :, : expected_width, : expected_height]

    if dim_ordering == 'tf':
        pool_out = pool_out.dimshuffle((0, 2, 3, 1))
    return pool_out


def spike_pool_2d(inputs, ds, ignore_border=None, st=None, padding=(0, 0),
                  mode='max'):
    """Downscale the input by a specified factor

    Takes as input a N-D tensor, where N >= 2. It downscales the input image by
    the specified factor, by keeping only the maximum value of non-overlapping
    patches of size (ds[0],ds[1])

    Parameters
    ----------
    inputs : list[N-D theano tensors of input images]
        Input images. Max pooling will be done over the 2 last dimensions.
    ds : tuple of length 2
        Factor by which to downscale (vertical ds, horizontal ds).
        (2,2) will halve the image in each dimension.
    ignore_border : bool (default None, will print a warning and set to False)
        When True, (5,5) input with ds=(2,2) will generate a (2,2) output.
        (3,3) otherwise.
    st : tuple of two ints
        Stride size, which is the number of shifts over rows/cols to get the
        next pool region. If st is None, it is considered equal to ds
        (no overlap on pooling regions).
    padding : tuple of two ints
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
            " `ds == st and padding == (0, 0) and mode == 'max'`."
            " Otherwise, the convolution will be executed on CPU.",
            stacklevel=2)
        ignore_border = False
    if x.ndim == 4:
        op = SpikePool(ds, ignore_border, st=st, padding=padding, mode=mode)
        output = op(inputs)
        return output
    else:
        raise NotImplementedError


class SpikePool(theano.Op):
    """
    For N-dimensional tensors, consider that the last two dimensions span
    images. This Op downsamples these images by taking the max, sum or average
    over different patch.

    The constructor takes the max, sum or average or different input patches.

    Parameters
    ----------
    ds : list or tuple of two ints
        Downsample factor over rows and column.
        ds indicates the pool region size.
    ignore_border : bool
        If ds doesn't divide imgshape, do we include an extra row/col
        of partial downsampling (False) or ignore it (True).
    st : list or tuple of two ints or None
        Stride size, which is the number of shifts over rows/cols to get the
        next pool region. If st is None, it is considered equal to ds
        (no overlap on pooling regions).
    padding: tuple of two ints
        (pad_h, pad_w), pad zeros to extend beyond four borders of the images,
        pad_h is the size of the top and bottom margins, and pad_w is the size
        of the left and right margins.
    mode : {'max', 'sum', 'average_inc_pad', 'average_exc_pad'}
        ('average_inc_pad' excludes the padding from the count,
        'average_exc_pad' include it)

    """

    __props__ = ('ds', 'ignore_border', 'st', 'padding', 'mode')

    def __init__(self, ds, ignore_border=False, st=None, padding=(0, 0),
                 mode='max'):
        super(SpikePool, self).__init__()
        self.ds = tuple(ds)
        if st is None:
            st = ds
        assert isinstance(st, (tuple, list))
        self.st = tuple(st)
        self.ignore_border = ignore_border
        self.padding = tuple(padding)
        if self.padding != (0, 0) and not ignore_border:
            raise NotImplementedError(
                'padding works only with ignore_border=True')
        if self.padding[0] >= self.ds[0] or self.padding[1] >= self.ds[1]:
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

    def make_node(self, x):
        """

        Parameters
        ----------
        x :

        Returns
        -------

        """

        for i in range(len(x)):
            if x[i].type.ndim != 4:
                raise TypeError()
            x[i] = t.as_tensor_variable(x[i])
        # If the input shape are broadcastable we can have 0 in the output shape
        broad = x[0].broadcastable[:2] + (False, False)
        out = t.TensorType(x[0].dtype, broad)
        return theano.gof.Apply(self, x, [out()])

    def perform(self, node, inp, out, **kwargs):
        """Perform pooling operation on spikes.

        Parameters
        ----------

        **kwargs :
        out :
        inp :
        node :
        """

        xr = inp[0]  # Presynaptic spike-rates
        xs = inp[1]  # Presynaptic spikes
        z, = out
        if len(xr.shape) != 4:
            raise NotImplementedError(
                'Pool requires 4D input for now')
        z_shape = pool.Pool.out_shape(xr.shape, self.ds, self.ignore_border,
                                      self.st, self.padding)
        if (z[0] is None) or (z[0].shape != z_shape):
            z[0] = np.zeros(z_shape, dtype=xr.dtype)
        zz = z[0]
        # number of pooling output rows
        pr = zz.shape[-2]
        # number of pooling output cols
        pc = zz.shape[-1]
        ds0, ds1 = self.ds
        st0, st1 = self.st
        pad_h = self.padding[0]
        pad_w = self.padding[1]
        img_rows = xr.shape[-2] + 2 * pad_h
        img_cols = xr.shape[-1] + 2 * pad_w
        inc_pad = self.mode == 'average_inc_pad'

        # pad the image
        if self.padding != (0, 0):
            yr = np.zeros(
                (xr.shape[0], xr.shape[1], img_rows, img_cols),
                dtype=xr.dtype)
            yr[:, :, pad_h:(img_rows - pad_h), pad_w:(img_cols - pad_w)] = xr
            ys = np.zeros(
                (xs.shape[0], xs.shape[1], img_rows, img_cols),
                dtype=xs.dtype)
            ys[:, :, pad_h:(img_rows - pad_h), pad_w:(img_cols - pad_w)] = xs
        else:
            yr = xr
            ys = xs

        for n in range(xr.shape[0]):
            for j in range(xr.shape[1]):
                for r in range(pr):
                    row_st = r * st0
                    row_end = min(row_st + ds0, img_rows)
                    if not inc_pad:
                        row_st = max(row_st, self.padding[0])
                        row_end = min(row_end, xr.shape[-2] + pad_h)
                    for c in range(pc):
                        col_st = c * st1
                        col_end = min(col_st + ds1, img_cols)
                        if not inc_pad:
                            col_st = max(col_st, self.padding[1])
                            col_end = min(col_end, xr.shape[-1] + pad_w)
                        rate_patch = yr[n, j, row_st:row_end, col_st:col_end]
                        # if not rate_patch.any():
                        #     # Need to prevent the layer to output a spike at
                        #     # index 0 if all rates are equally zero.
                        #     continue
                        spike_patch = ys[n, j, row_st:row_end, col_st:col_end]
                        # max_rates = rate_patch == np.max(rate_patch)
                        # if (spike_patch * max_rates).any():
                        #     zz[n, j, r, c] = settings['v_thresh']
                        max_rate_idx = np.argmax(rate_patch)  # flattens patch
                        if spike_patch.flatten()[max_rate_idx]:
                            zz[n, j, r, c] = settings['v_thresh']

custom_layers = {'SpikeFlatten': SpikeFlatten,
                 'SpikeDense': SpikeDense,
                 'SpikeConvolution2D': SpikeConvolution2D,
                 'SpikeAveragePooling2D': SpikeAveragePooling2D,
                 'SpikeMaxPooling2D': SpikeMaxPooling2D,
                 'SpikeMerge': SpikeMerge}
