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
from keras import backend as k
from keras.layers import Convolution2D
from keras.layers import Dense, Flatten, AveragePooling2D, MaxPooling2D
from snntoolbox.config import settings
from theano.tensor.shared_randomstreams import RandomStreams
from theano.tensor.signal import pool

standard_library.install_aliases()

rng = RandomStreams()


def float_x(x):
    """Return array in floatX settings of Theano."""

    return [np.asarray(i, dtype=theano.config.floatX) for i in x]


def shared_x(x, dtype=theano.config.floatX, name=None):
    """Make array as shared array."""

    return theano.shared(np.asarray(x, dtype=dtype), name=name)


def shared_zeros(shape, dtype=theano.config.floatX, name=None):
    """Make shared zeros array."""

    return shared_x(np.zeros(shape), dtype=dtype, name=name)


def on_gpu():
    """Check if running on GPU board."""

    return theano.config.device[:3] == 'gpu'


if on_gpu():
    from theano.sandbox.cuda import dnn


def update_neurons(self, time, updates):
    """Update neurons according to activation function."""

    if hasattr(self, 'activation_str'):
        if self.activation_str == 'softmax':
            output_spikes = softmax_activation(self, time, updates)
        elif self.activation_str == 'binary_sigmoid':
            output_spikes = binary_sigmoid_activation(self, time, updates)
        elif self.activation_str == 'binary_tanh':
            output_spikes = binary_tanh_activation(self, time, updates)
        else:
            output_spikes = linear_activation(self, time, updates)
    else:
        output_spikes = linear_activation(self, time, updates)

    # Store refractory
    new_refractory = t.set_subtensor(
        self.refrac_until[output_spikes.nonzero()], time + self.tau_refrac)
    updates.append((self.refrac_until, new_refractory))

    updates.append((self.spiketrain, output_spikes * (time + settings['dt'])))
    updates.append((self.spikecounts, self.spikecounts + output_spikes))

    axes = np.arange(len(self.output_shape))
    reduction_axes = tuple(axes[1:])
    updates.append((self.total_spike_count, t.sum(self.spikecounts,
                                                  reduction_axes)))

    if settings['online_normalization']:
        updates.append((self.max_spikerate,
                        t.max(self.spikecounts + output_spikes) *
                        settings['dt'] / (time + settings['dt'])))

    if hasattr(self, 'spikerate'):
        if settings['maxpool_type'] == 'fir_max':
            updates.append((self.spikerate,
                            self.spikerate + (output_spikes - self.spikerate) /
                            ((time + settings['dt']) / settings['dt'])))
        elif settings['maxpool_type'] == 'fir_max':
            updates.append((self.spikerate,
                            self.spikerate + output_spikes /
                            ((time + settings['dt']) / settings['dt'])))
        #    updates.append((self.spikerate,
        #                    (self.spikecounts+output_spikes) *
        #                    settings['dt'] / (time + settings['dt'])))
        elif settings['maxpool_type'] == 'exp_max':
            updates.append((self.spikerate,
                            self.spikerate + output_spikes /
                            2. ** ((time + settings['dt']) / settings['dt'])))
    return output_spikes


def update_payload(self, residuals, idxs):
    """Update payloads.

    Uses the residual of the membrane potential after spike.
    """

    payloads = t.set_subtensor(
        self.payloads[idxs], residuals[idxs] - self.payloads_sum[idxs])
    payloads_sum = t.set_subtensor(
        self.payloads_sum[idxs], self.payloads_sum[idxs] + self.payloads[idxs])
    return payloads, payloads_sum


def binary_sigmoid_activation(self, time, updates):
    """Binary sigmoid activation."""

    # Destroy impulse if in refractory period
    masked_imp = t.set_subtensor(
        self.impulse[t.nonzero(self.refrac_until > time)], 0.)

    # Add impulse
    new_mem = self.mem + masked_imp

    # Store spiking
    output_spikes = t.gt(new_mem, 0)

    spike_idxs = output_spikes.nonzero()

    # Reset neurons
    new_and_reset_mem = t.set_subtensor(new_mem[spike_idxs], 0.)

    updates.append((self.mem, new_and_reset_mem))

    return output_spikes


def binary_tanh_activation(self, time, updates):
    """Binary tanh activation."""

    # Destroy impulse if in refractory period
    masked_imp = t.set_subtensor(
        self.impulse[t.nonzero(self.refrac_until > time)], 0.)

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

    updates.append((self.mem, new_and_reset_mem))

    return signed_spikes


def linear_activation(self, time, updates):
    """Linear activation."""

    # Destroy impulse if in refractory period
    masked_imp = t.set_subtensor(
        self.impulse[t.nonzero(self.refrac_until > time)], 0.)

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

    updates.append((self.mem, new_and_reset_mem))

    if settings['payloads']:
        residuals = t.inc_subtensor(new_mem[spike_idxs], -self.v_thresh)
        payloads, payloads_sum = update_payload(self, residuals, spike_idxs)
        updates.append((self.payloads, payloads))
        updates.append((self.payloads_sum, payloads_sum))

    return output_spikes


def softmax_activation(self, time, updates):
    """Softmax activation."""

    # Destroy impulse if in refractory period
    masked_imp = t.set_subtensor(
        self.impulse[t.nonzero(self.refrac_until > time)], 0.)
    # Add impulse
    new_mem = self.mem + masked_imp
    # Store spiking
    spiking_samples = t.le(rng.uniform([settings['batch_size'], 1]),
                           settings['softmax_clockrate'] * settings[
                               'dt'] / 1000.)
    spiking_neurons = t.repeat(spiking_samples, 10, axis=1)
    activ = t.nnet.softmax(new_mem)
    max_activ = t.max(activ, axis=1, keepdims=True)
    output_spikes = t.eq(activ, max_activ).astype('float32')
    output_spikes = t.set_subtensor(
        output_spikes[t.eq(spiking_neurons, 0).nonzero()], 0.)
    new_and_reset_mem = t.set_subtensor(new_mem[spiking_neurons.nonzero()], 0.)
    updates.append((self.mem, new_and_reset_mem))

    return output_spikes


def reset(self):
    """Reset."""

    if self.inbound_nodes[0].inbound_layers:
        reset(self.inbound_nodes[0].inbound_layers[0])
    self.mem.set_value(float_x(np.zeros(self.output_shape)))
    self.refrac_until.set_value(float_x(np.zeros(self.output_shape)))
    self.spiketrain.set_value(float_x(np.zeros(self.output_shape)))
    self.spikecounts.set_value(float_x(np.zeros(self.output_shape)))
    self.total_spike_count.set_value(float_x(np.zeros(settings['batch_size'])))

    if settings['payloads']:
        self.payloads.set_value(float_x(np.zeros(self.output_shape)))
        self.payloads_sum.set_value(float_x(np.zeros(self.output_shape)))

    if settings['online_normalization']:
        self.max_spikerate.set_value(0.0)
        self.v_thresh.set_value(settings['v_thresh'])

    if hasattr(self, 'spikerate'):
        self.spikerate.set_value(float_x(np.zeros(self.output_shape)))


def get_input(self):
    """Get input."""

    if self.inbound_nodes[0].inbound_layers:
        if 'input' in self.inbound_nodes[0].inbound_layers[0].name:
            previous_output = self.input
        else:
            previous_output = \
                self.inbound_nodes[0].inbound_layers[0].get_output()
    else:
        previous_output = k.placeholder(shape=self.input_shape)
    return previous_output, get_time(self), get_updates(self)


def get_time(self):
    """Get time."""

    if hasattr(self, 'time_var'):
        return self.time_var
    elif self.inbound_nodes[0].inbound_layers:
        return get_time(self.inbound_nodes[0].inbound_layers[0])
    else:
        raise Exception("Layer is not connected and is not an input layer.")


def get_updates(self):
    """Get updates."""

    if self.inbound_nodes[0].inbound_layers:
        return self.inbound_nodes[0].inbound_layers[0].updates
    else:
        return []


def init_neurons(self, v_thresh=1.0, tau_refrac=0.0, **kwargs):
    """Init neurons."""

    # The neurons in the spiking layer cannot be initialized until the layer
    # has been initialized and connected to the network. Otherwise
    # 'output_shape' is not known (obtained from previous layer), and
    # the 'input' attribute will not be overwritten by the layer's __init__.
    init_layer(self, self, v_thresh, tau_refrac)
    if 'time_var' in kwargs:
        input_layer = self.inbound_nodes[0].inbound_layers[0]
        input_layer.time_var = kwargs['time_var']
        init_layer(self, input_layer, v_thresh, tau_refrac)


def init_layer(self, layer, v_thresh, tau_refrac):
    """Init layer."""

    layer.v_thresh = theano.shared(
        np.asarray(v_thresh, dtype=theano.config.floatX), 'v_thresh')
    layer.tau_refrac = tau_refrac
    layer.refrac_until = shared_zeros(self.output_shape)
    layer.mem = shared_zeros(self.output_shape)
    layer.spiketrain = shared_zeros(self.output_shape)
    layer.spikecounts = shared_zeros(self.output_shape)
    layer.total_spike_count = shared_zeros(settings['batch_size'])
    if settings['payloads']:
        layer.payloads = shared_zeros(self.output_shape)
        layer.payloads_sum = shared_zeros(self.output_shape)
    layer.updates = []
    layer.layer_type = layer.__class__.__name__

    if settings['online_normalization']:
        layer.max_spikerate = theano.shared(np.asarray([0.0], 'float32'))

    if layer.layer_type == "SpikeMaxPooling2D":
        prev_layer = self.inbound_nodes[0].inbound_layers[0]
        prev_layer.spikerate = shared_zeros(self.output_shape)

    if len(layer.get_weights()) > 0:
        layer.W = k.variable(layer.get_weights()[0])
        layer.b = k.variable(layer.get_weights()[1])


def get_new_thresh(self, time):
    """Get new threshhold."""

    return theano.ifelse.ifelse(
        t.eq(time / settings['dt'] % settings['timestep_fraction'], 0) *
        t.gt(self.max_spikerate, settings['diff_to_min_rate'] / 1000) *
        t.gt(1 / settings['dt'] - self.max_spikerate,
             settings['diff_to_max_rate'] / 1000),
        self.max_spikerate, self.v_thresh)


class SpikeFlatten(Flatten):
    """Spike flatten layer."""

    def __init__(self, **kwargs):
        """Init function."""
        super(SpikeFlatten, self).__init__(**kwargs)
        self.updates = None
        if settings['payloads']:
            self.payloads = None
            self.payloads_sum = None

    def get_output(self):
        """Get output."""
        # Recurse
        inp, time, updates = get_input(self)
        reshaped_inp = t.reshape(inp, self.output_shape)
        if settings['payloads']:
            payloads = t.reshape(self.payloads, self.output_shape)
            payloads_sum = t.reshape(self.payloads_sum, self.output_shape)
            updates.append((self.payloads, payloads))
            updates.append((self.payloads_sum, payloads_sum))
        self.updates = updates
        return reshaped_inp

    def get_name(self):
        """Get class name."""
        return self.__class__.__name__


class SpikeDense(Dense):
    """Spike Dense layer."""

    def __init__(self, output_dim, weights=None, **kwargs):
        """Init function."""
        # Remove activation from kwargs before initializing superclass, in case
        # we are using a custom activation function that Keras doesn't
        # understand.
        self.activation_str = str(kwargs.pop('activation'))
        super(SpikeDense, self).__init__(output_dim=output_dim, weights=weights,
                                         **kwargs)
        self.v_thresh = self.tau_refrac = self.mem = self.spiketrain = None
        self.impulse = self.spikecounts = self.total_spike_count = None
        self.updates = self.refrac_until = self.max_spikerate = None

    def get_output(self):
        """Get output."""

        # Recurse
        inp, time, updates = get_input(self)
        if settings['online_normalization']:
            # Modify threshold if firing rate of layer too low
            updates.append((self.v_thresh, get_new_thresh(self, time)))
        if settings['payloads']:
            # Add payload from previous layer
            prev_layer = self.inbound_nodes[0].inbound_layers[0]
            inp = add_payloads(prev_layer, inp)

        self.impulse = t.add(t.dot(inp, self.W), self.b)
        output_spikes = update_neurons(self, time, updates)
        self.updates = updates
        return t.cast(output_spikes, 'float32')

    def get_name(self):
        """Get class name."""
        return self.__class__.__name__


class SpikeConvolution2D(Convolution2D):
    """Spike 2D Convolution."""

    def __init__(self, nb_filter, nb_row, nb_col, weights=None,
                 border_mode='valid', subsample=(1, 1), filter_flip=True,
                 **kwargs):
        """Init function."""
        self.activation_str = str(kwargs.pop('activation'))
        super(SpikeConvolution2D, self).__init__(
            nb_filter=nb_filter, nb_row=nb_row, nb_col=nb_col, weights=weights,
            border_mode=border_mode, subsample=subsample, **kwargs)
        self.filter_flip = filter_flip
        self.v_thresh = self.tau_refrac = self.mem = self.spiketrain = None
        self.impulse = self.spikecounts = self.total_spike_count = None
        self.updates = self.refrac_until = self.max_spikerate = None

    def get_output(self):
        """Get output."""

        # Recurse
        inp, time, updates = get_input(self)
        if settings['payloads']:
            # Add payload from previous layer
            prev_layer = self.inbound_nodes[0].inbound_layers[0]
            inp = add_payloads(prev_layer, inp)

        if settings['online_normalization']:
            # Modify threshold if firing rate of layer too low
            updates.append((self.v_thresh, get_new_thresh(self, time)))

        # CALCULATE SYNAPTIC SUMMED INPUT
        border_mode = self.border_mode
        if on_gpu() and dnn.dnn_available():
            conv_mode = 'conv' if self.filter_flip else 'cross'
            if border_mode == 'same':
                assert (self.subsample == (1, 1))
                pad_x = (self.nb_row - self.subsample[0]) // 2
                pad_y = (self.nb_col - self.subsample[1]) // 2
                conv_out = dnn.dnn_conv(img=inp, kerns=self.W,
                                        border_mode=(pad_x, pad_y),
                                        conv_mode=conv_mode)
            else:
                conv_out = dnn.dnn_conv(img=inp, kerns=self.W,
                                        border_mode=border_mode,
                                        subsample=self.subsample,
                                        conv_mode=conv_mode)
        else:
            if border_mode == 'same':
                border_mode = 'full'
            conv_out = t.nnet.conv2d(inp, self.W, border_mode=border_mode,
                                     subsample=self.subsample,
                                     filter_flip=self.filter_flip)
            if self.border_mode == 'same':
                shift_x = (self.nb_row - 1) // 2
                shift_y = (self.nb_col - 1) // 2
                conv_out = conv_out[:, :, shift_x:inp.shape[2] + shift_x,
                                    shift_y:inp.shape[3] + shift_y]
        self.impulse = conv_out + k.reshape(self.b, (1, self.nb_filter, 1, 1))
        output_spikes = update_neurons(self, time, updates)
        self.updates = updates
        return t.cast(output_spikes, 'float32')

    def get_name(self):
        """Get class name."""

        return self.__class__.__name__


class SpikeAveragePooling2D(AveragePooling2D):
    """Average Pooling."""

    def __init__(self, pool_size=(2, 2), strides=None, border_mode='valid',
                 **kwargs):
        """Init average pooling."""

        super(SpikeAveragePooling2D, self).__init__(
            pool_size=pool_size, strides=strides, border_mode=border_mode,
            **kwargs)
        self.v_thresh = self.tau_refrac = self.mem = self.spiketrain = None
        self.impulse = self.spikecounts = self.total_spike_count = None
        self.updates = self.refrac_until = None

    def get_output(self):
        """Get output."""

        # Recurse
        inp, time, updates = get_input(self)

        if settings['payloads']:
            # Add payload from previous layer
            prev_layer = self.inbound_nodes[0].inbound_layers[0]
            inp = add_payloads(prev_layer, inp)

        self.impulse = k.pool2d(inp, self.pool_size, self.strides,
                                self.border_mode, pool_mode='avg')

        output_spikes = update_neurons(self, time, updates)
        self.updates = updates
        return t.cast(output_spikes, 'float32')

    def get_name(self):
        """Get class name."""

        return self.__class__.__name__


class SpikeMaxPooling2D(MaxPooling2D):
    """Max Pooling."""

    def __init__(self, pool_size=(2, 2), strides=None, border_mode='valid',
                 **kwargs):
        """Init function."""

        self.ignore_border = True if border_mode == 'valid' else False
        if 'binary' in settings['maxpool_type']:
            self.activation_str = settings['maxpool_type']
        super(SpikeMaxPooling2D, self).__init__(
            pool_size=pool_size, strides=strides,
            border_mode=border_mode, **kwargs)
        self.v_thresh = self.tau_refrac = self.mem = self.spiketrain = None
        self.impulse = self.spikecounts = self.total_spike_count = None
        self.updates = self.refrac_until = None

    def get_output(self):
        """Get output."""

        # Recurse
        inp, time, updates = get_input(self)

        if settings['payloads']:
            # Add payload from previous layer
            prev_layer = self.inbound_nodes[0].inbound_layers[0]
            inp = add_payloads(prev_layer, inp)

        if 'binary' in settings['maxpool_type']:
            self.impulse = k.pool2d(inp, self.pool_size, self.strides,
                                    self.border_mode, pool_mode='max')
        elif settings['maxpool_type'] in ["avg_max", "fir_max", "exp_max"]:
            spikerate = self.inbound_nodes[0].inbound_layers[0].spikerate \
                if self.inbound_nodes[0].inbound_layers \
                else k.placeholder(shape=self.input_shape)
            max_idx = pool_same_size(spikerate, patch_size=self.pool_size,
                                     ignore_border=self.ignore_border,
                                     st=self.strides)
            self.impulse = k.pool2d(inp * max_idx, self.pool_size, self.strides,
                                    self.border_mode, pool_mode='max')
        else:
            print("Wrong max pooling type, "
                  "falling back on Average Pooling instead.")
            self.impulse = k.pool2d(inp, self.pool_size, self.strides,
                                    self.border_mode, pool_mode='avg')

        output_spikes = update_neurons(self, time, updates)
        self.updates = updates
        return t.cast(output_spikes, 'float32')

    def get_name(self):
        """Get class name."""

        return self.__class__.__name__


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
    return t.cast(outs, 'float32')


def add_payloads(prev_layer, input_spikes):
    """Get payloads from previous layer."""

    # Get only payloads of those pre-synaptic neurons that spiked
    payloads = t.set_subtensor(
        prev_layer.payloads[t.nonzero(t.eq(input_spikes, 0.))], 0.)
    print("Using spikes with payloads from layer {}".format(prev_layer.name))
    return t.add(input_spikes, payloads)


custom_layers = {'SpikeFlatten': SpikeFlatten,
                 'SpikeDense': SpikeDense,
                 'SpikeConvolution2D': SpikeConvolution2D,
                 'SpikeAveragePooling2D': SpikeAveragePooling2D,
                 'SpikeMaxPooling2D': SpikeMaxPooling2D}
