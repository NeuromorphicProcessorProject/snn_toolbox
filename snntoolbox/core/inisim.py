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

import numpy as np
import theano
import theano.tensor as t
from future import standard_library
import inspect
from keras import backend as k
from keras.layers import Convolution2D, Merge
from keras.layers import Dense, Flatten, AveragePooling2D, MaxPooling2D
import keras.activations as k_activ
from keras.engine.topology import to_list
from snntoolbox.config import settings
from theano.tensor.shared_randomstreams import RandomStreams
from theano.tensor.signal import pool
from typing import Union

standard_library.install_aliases()

rng = RandomStreams()

floatX = theano.config.floatX


def float_x(x):
    """Return array in floatX settings of Theano."""

    return [np.asarray(i, floatX) for i in x]


def shared_x(x, dtype=floatX, name=None):
    """Make array as shared array."""

    return theano.shared(np.asarray(x, dtype), name)


def shared_zeros(shape, dtype=floatX, name=None):
    """Make shared zeros array."""

    return shared_x(np.zeros(shape), dtype, name)


def update_neurons(self, x):
    """Update neurons according to activation function."""

    if hasattr(self, 'activation_str'):
        if self.activation_str == 'softmax':
            output_spikes = softmax_activation(self, x)
        elif self.activation_str == 'binary_sigmoid':
            output_spikes = binary_sigmoid_activation(self, x)
        elif self.activation_str == 'binary_tanh':
            output_spikes = binary_tanh_activation(self, x)
        else:
            output_spikes = linear_activation(self, x)
    else:
        output_spikes = linear_activation(self, x)

    # Store refractory
    if settings['tau_refrac'] > 0:
        new_refractory = t.set_subtensor(
            self.refrac_until[output_spikes.nonzero()],
            self.time + self.tau_refrac)
        add_updates(self, (self.refrac_until, new_refractory), x)

    if settings['verbose'] > 1 or settings['online_normalization']:
        add_updates(self, (self.spikecounts, self.spikecounts + output_spikes),
                    x)

    # This update of the spiketrains does not seem to work (add_updates in
    # principle? Would affect all other variables, like mem, which would explain
    # why activity in the first layer is so low and there is nothing propagated
    # beyond the first layer.). We get singe spikes in the first layer feature
    # maps at each time step, but they don't show up in the spiketrain variable.
    if settings['verbose'] > 1:
        add_updates(self, (self.spiketrain, output_spikes *
                           (self.time + settings['dt'])), x)
        reduction_axes = tuple(np.arange(self.mem.ndim)[1:])
        add_updates(self, (self.total_spike_count,
                           t.sum(self.spikecounts, reduction_axes)), x)

    if settings['online_normalization']:
        add_updates(self, (self.max_spikerate, t.max(self.spikecounts) *
                           settings['dt'] / (self.time + settings['dt'])), x)

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


def binary_sigmoid_activation(self, x):
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

    add_updates(self, (self.mem, new_and_reset_mem), x)

    return output_spikes


def binary_tanh_activation(self, x):
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

    add_updates(self, (self.mem, new_and_reset_mem), x)

    return signed_spikes


def linear_activation(self, x):
    """Linear activation."""

    # Destroy impulse if in refractory period
    masked_imp = self.impulse if settings['tau_refrac'] == 0 else \
        t.set_subtensor(self.impulse[t.nonzero(self.refrac_until > self.time)],
                        0.)

    # Add impulse
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
    else:  # settings['reset'] == 'Reset to zero':
        new_and_reset_mem = t.set_subtensor(new_mem[spike_idxs], 0.)

    add_updates(self, (self.mem, new_and_reset_mem), x)

    if settings['payloads']:
        residuals = t.inc_subtensor(new_mem[spike_idxs], -self.v_thresh)
        payloads, payloads_sum = update_payload(self, residuals, spike_idxs)
        add_updates(self, (self.payloads, payloads), x)
        add_updates(self, (self.payloads_sum, payloads_sum), x)

    return output_spikes


def softmax_activation(self, x):
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
    add_updates(self, (self.mem, new_and_reset_mem), x)

    return output_spikes


def reset_spikevars(self):
    """Reset variables present in spiking layers."""

    self.mem.set_value(np.zeros(self.output_shape, floatX))
    if settings['tau_refrac'] > 0:
        self.refrac_until.set_value(np.zeros(self.output_shape, floatX))
    if settings['verbose'] > 1:
        self.spiketrain.set_value(np.zeros(self.output_shape, floatX))
        self.total_spike_count.set_value(np.zeros(settings['batch_size'],
                                                  floatX))
    if settings['verbose'] > 1 or settings['online_normalization']:
        self.spikecounts.set_value(np.zeros(self.output_shape, floatX))
    if settings['payloads']:
        self.payloads.set_value(np.zeros(self.output_shape, floatX))
        self.payloads_sum.set_value(np.zeros(self.output_shape, floatX))
    if settings['online_normalization']:
        self.max_spikerate.set_value(0.0)
        self.v_thresh.set_value(settings['v_thresh'])


def init_neurons(self, input_shape, tau_refrac=0.0):
    """Init layer neurons."""

    output_shape = self.get_output_shape_for(input_shape)
    self.v_thresh = shared_x(settings['v_thresh'], name='v_thresh')
    self.tau_refrac = tau_refrac
    self.mem = theano.shared(np.zeros(output_shape, floatX))
    self.layer_type = self.__class__.__name__
    # To save memory and computations, allocate only where needed:
    if settings['tau_refrac'] > 0:
        self.refrac_until = theano.shared(np.zeros(output_shape, floatX))
    if settings['verbose'] > 1:
        self.spiketrain = theano.shared(np.zeros(output_shape, floatX))
        self.total_spike_count = theano.shared(np.zeros(settings['batch_size'],
                                                        floatX))
    if settings['verbose'] > 1 or settings['online_normalization']:
        self.spikecounts = theano.shared(np.zeros(output_shape, floatX))
    if settings['payloads']:
        self.payloads = theano.shared(np.zeros(output_shape, floatX))
        self.payloads_sum = theano.shared(np.zeros(output_shape, floatX))
    if settings['online_normalization']:
        self.max_spikerate = theano.shared(np.array([0.0], floatX))


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

    return self.time.get_value()[0] if hasattr(self, 'time') else None


def set_time(self, time):
    """Set simulation time variable.

    Parameters
    ----------

    self: SpikeLayer
        SpikeLayer derived from keras.layers.Layer.
    time: float
        Current simulation time.
    """

    self.time.set_value([time])


def add_updates(self, updates, inputs):
    """Update self.updates.
    This is taken from a development-version of Keras. Might be able to remove
    it with the next official version. (27.11.16)"""

    if not hasattr(self, 'updates'):
        self.updates = []
    try:
        self.updates += updates
    except AttributeError:
        pass
    # Update self._per_input_updates
    if not hasattr(self, '_per_input_updates'):
        self._per_input_updates = {}
    inputs = to_list(inputs)
    updates = to_list(updates)
    inputs_hash = ', '.join([str(abs(id(x))) for x in inputs])
    if inputs_hash not in self._per_input_updates:
        self._per_input_updates[inputs_hash] = []
    self._per_input_updates[inputs_hash] += updates


def pool_same_size(data_in, patch_size, ignore_border=True, st=None,
                   padding=(0, 0)):
    """Max-pooling in same size.

    The indices of maximum values are 1s, else are 0s.

    Parameters
    ----------

    data_in: 4-D tensor.
        input images. Max-pooling will be done over the 2 last dimensions.
    patch_size: tuple
        with length 2 (patch height, patch width)
    ignore_border: bool
        When True, (5,5) input with ds=(2,2) will generate a (2,2) output.
        (3,3) otherwise.
    st: tuple
        Stride size, which is the number of shifts over rows/cols to get the
        next pool region. If st is None, it is considered equal to ds
        (no overlap on pooling regions).
    padding: tuple
        (pad_h, pad_w) pad zeros to extend beyond four borders of the
        images, pad_h is the size of the top and bottom margins, and
        pad_w is the size of the left and right margins.
    """

    output = pool.Pool(ds=patch_size, ignore_border=ignore_border,
                       st=st, padding=padding, mode="max")(data_in)
    outs = pool.MaxPoolGrad(ds=patch_size, ignore_border=ignore_border,
                            st=st, padding=padding)(data_in, output,
                                                    output) > 0.
    return t.cast(outs, floatX)


def add_payloads(prev_layer, input_spikes):
    """Get payloads from previous layer."""

    # Get only payloads of those pre-synaptic neurons that spiked
    payloads = t.set_subtensor(
        prev_layer.payloads[t.nonzero(t.eq(input_spikes, 0.))], 0.)
    print("Using spikes with payloads from layer {}".format(prev_layer.name))
    return t.add(input_spikes, payloads)


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

    def __init__(self, **kwargs):
        """Init function."""

        super(SpikeFlatten, self).__init__(**kwargs)
        self.updates = []
        self._per_input_updates = {}
        if settings['payloads']:
            self.payloads = None
            self.payloads_sum = None

    def call(self, x, mask=None):
        """Layer functionality."""

        if settings['payloads']:
            payloads = t.reshape(self.payloads, self.output_shape)
            payloads_sum = t.reshape(self.payloads_sum, self.output_shape)
            add_updates(self, (self.payloads, payloads), x)
            add_updates(self, (self.payloads_sum, payloads_sum), x)
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
        # Remove activation from kwargs before initializing superclass, in case
        # we are using a custom activation function that Keras doesn't
        # understand.
        self.activation_str = str(kwargs['activation'])
        activs = [a[0] for a in inspect.getmembers(k_activ, inspect.isfunction)]
        if self.activation_str not in activs:
            kwargs.pop('activation')
            if not settings['convert']:
                print("WARNING: It seems you have restored a previously "
                      "converted SNN from disk, which uses an activation "
                      "function unknown to Keras. This custom function could "
                      "not be saved and reloaded. Falling back on 'linear' "
                      "activation. Convert from scratch before simulating to "
                      "use custom function {}.".format(self.activation_str))
        super(SpikeDense, self).__init__(output_dim, **kwargs)
        self.layer_type = self.class_name
        self.tau_refrac = kwargs['tau_refrac'] if 'tau_refrac' in kwargs else 0.
        self.v_thresh = None
        self.updates = []
        self._per_input_updates = {}
        self.mem = self.spiketrain = self.impulse = self.spikecounts = None
        self.total_spike_count = self.refrac_until = self.max_spikerate = None
        self.time = k.zeros(1, name='time')

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

    def call(self, x, mask=None):
        """Layer functionality."""

        inp = x

        if settings['online_normalization']:
            # Modify threshold if firing rate of layer too low
            add_updates(self, (self.v_thresh, get_new_thresh(self)), x)
        if settings['payloads']:
            # Add payload from previous layer
            inp = add_payloads(self.inbound_nodes[0].inbound_layers[0], inp)

        self.impulse = super(SpikeDense, self).call(inp)
        return update_neurons(self, x)

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
        self.activation_str = str(kwargs.pop('activation'))
        super(SpikeConvolution2D, self).__init__(nb_filter, nb_row, nb_col,
                                                 **kwargs)
        self.layer_type = self.class_name
        self.filter_flip = filter_flip
        self.tau_refrac = kwargs['tau_refrac'] if 'tau_refrac' in kwargs else 0.
        self.v_thresh = None
        self.updates = []
        self._per_input_updates = {}
        self.mem = self.spiketrain = self.impulse = self.spikecounts = None
        self.total_spike_count = self.refrac_until = self.max_spikerate = None
        self.time = k.zeros(1, name='time')

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

    def call(self, x, mask=None):
        """Layer functionality."""

        inp = x

        if settings['payloads']:
            # Add payload from previous layer
            inp = add_payloads(self.inbound_nodes[0].inbound_layers[0], inp)

        if settings['online_normalization']:
            # Modify threshold if firing rate of layer too low
            add_updates(self, (self.v_thresh, get_new_thresh(self)), x)

        self.impulse = super(SpikeConvolution2D, self).call(inp)
        return update_neurons(self, x)

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
        self.updates = []
        self._per_input_updates = {}
        self.mem = self.spiketrain = self.impulse = self.spikecounts = None
        self.total_spike_count = self.refrac_until = self.max_spikerate = None
        self.time = k.zeros(1, name='time')

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

        inp = x

        if settings['payloads']:
            # Add payload from previous layer
            inp = add_payloads(self.inbound_nodes[0].inbound_layers[0], inp)

        self.impulse = super(SpikeAveragePooling2D, self).call(inp)
        return update_neurons(self, x)

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
        self.updates = []
        self._per_input_updates = {}
        self.time = k.zeros(1, name='time')
        self.spikerate = None
        self.mem = self.spiketrain = self.impulse = self.spikecounts = None
        self.total_spike_count = self.refrac_until = self.max_spikerate = None

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
        self.spikerate = shared_zeros(input_shape)

    def call(self, x, mask=None):
        """Layer functionality."""

        inp = x

        t_inv = settings['dt'] / (self.time + settings['dt'])
        if settings['maxpool_type'] == 'avg_max':
            add_updates(self, (self.spikerate, self.spikerate +
                               (x - self.spikerate) * t_inv), x)
        elif settings['maxpool_type'] == 'fir_max':
            add_updates(self, (self.spikerate, self.spikerate + x * t_inv), x)
            # add_updates(self, (self.spikerate, (self.spikerate * self.time /
            #                                     settings['dt'] + x) * t_inv),
            #                    x)
            # add_updates(self, (self.spikerate, self.spikecounts * t_inv), x)
        elif settings['maxpool_type'] == 'exp_max':
            add_updates(self, (self.spikerate, self.spikerate +
                               x / 2. ** (1 / t_inv)), x)

        if settings['payloads']:
            # Add payload from previous layer
            inp = add_payloads(self.inbound_nodes[0].inbound_layers[0], inp)

        if 'binary' in settings['maxpool_type']:
            self.impulse = super(SpikeMaxPooling2D, self).call(inp)
        elif settings['maxpool_type'] in ["avg_max", "fir_max", "exp_max"]:
            max_idx = pool_same_size(self.spikerate, self.pool_size,
                                     self.ignore_border, self.strides)
            self.impulse = super(SpikeMaxPooling2D, self).call(t.mul(inp,
                                                                     max_idx))
        else:
            print("Wrong max pooling type, "
                  "falling back on Average Pooling instead.")
            self.impulse = k.pool2d(inp, self.pool_size, self.strides,
                                    self.border_mode, pool_mode='avg')

        return update_neurons(self, x)

    def reset(self):
        """Reset layer variables."""

        reset_spikevars(self)
        self.spikerate.set_value(np.zeros(self.input_shape, floatX))

    @property
    def class_name(self):
        """Get class name."""

        return self.__class__.__name__


custom_layers = {'SpikeFlatten': SpikeFlatten,
                 'SpikeDense': SpikeDense,
                 'SpikeConvolution2D': SpikeConvolution2D,
                 'SpikeAveragePooling2D': SpikeAveragePooling2D,
                 'SpikeMaxPooling2D': SpikeMaxPooling2D,
                 'SpikeMerge': SpikeMerge}
