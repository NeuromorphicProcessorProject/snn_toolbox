# -*- coding: utf-8 -*-
"""INI spiking neuron simulator.

A collection of helper functions, including spiking layer classes derived from
Keras layers, which were used to implement our own IF spiking simulator.

Not needed when converting and running the SNN in pyNN.

Created on Tue Dec  8 10:41:10 2015

@author: rbodo
"""

# For compatibility with python2
from __future__ import print_function, unicode_literals
from __future__ import division, absolute_import
from future import standard_library
from builtins import super

import numpy as np
import theano
import theano.tensor as T
from theano.tensor.signal import pool
from theano.tensor.shared_randomstreams import RandomStreams
from keras.layers.core import Dense, Flatten
from keras.layers.convolutional import AveragePooling2D, MaxPooling2D
from keras.layers.convolutional import Convolution2D
from keras import backend as K
from snntoolbox.config import settings

standard_library.install_aliases()

rng = RandomStreams()


def floatX(X):
    """Return array in floatX settings of Theano."""
    return [np.asarray(x, dtype=theano.config.floatX) for x in X]


def sharedX(X, dtype=theano.config.floatX, name=None):
    """Make array as shared array."""
    return theano.shared(np.asarray(X, dtype=dtype), name=name)


def shared_zeros(shape, dtype=theano.config.floatX, name=None):
    """Make shared 0s array."""
    return sharedX(np.zeros(shape), dtype=dtype, name=name)


def on_gpu():
    """Check if running on GPU board."""
    return theano.config.device[:3] == 'gpu'

if on_gpu():
    from theano.sandbox.cuda import dnn


def update_neurons(self, impulse, time, updates):
    """update neurons according to activation function."""
    if 'activation' in self.get_config() and \
            self.get_config()['activation'] == 'softmax':
        output_spikes = softmax_activation(self, impulse, time, updates)
    else:
        output_spikes = linear_activation(self, impulse, time, updates)
    updates.append((self.spiketrain, output_spikes * time))
    if settings['online_normalization']:
        updates.append((self.spikecounts, self.spikecounts + output_spikes))
        updates.append((self.max_spikerate,
                        T.max(self.spikecounts) / (time + settings['dt'])))
        if self.layer_type in ["MaxPool2DReLU", "SpikeConv2DReLU"]:
            if settings["maxpool_type"] == "avg_max":
                updates.append((self.avg_spikerate,
                                self.spikecounts / (time + settings['dt'])))
            elif settings["maxpool_type"] == "fir_max":
                updates.append((self.fir_spikerate,
                                self.fir_spikerate + output_spikes /
                                (time + settings['dt'])))
            elif settings["maxpool_type"] == "exp_max":
                updates.append((self.exp_spikerate,
                                self.exp_spikerate + output_spikes /
                                2.**(time + settings['dt'])))
    return output_spikes


def linear_activation(self, impulse, time, updates):
    """Linear activation."""
    # Destroy impulse if in refractory period
    masked_imp = T.set_subtensor(
        impulse[(self.refrac_until > time).nonzero()], 0.)
    # Add impulse
    new_mem = self.mem + masked_imp
    # Store spiking
    output_spikes = T.ge(new_mem, self.v_thresh)
    # At spike, reduce membrane potential by one instead of resetting to zero,
    # so that no information stored in membrane potential is lost. This reduces
    # the variance in the spikerate-activation correlation plot for activations
    # greater than 0.5.
    if settings['reset'] == 'Reset to zero':
        new_and_reset_mem = T.set_subtensor(
            new_mem[output_spikes.nonzero()], 0)
    elif settings['reset'] == 'Reset by subtraction':
        new_and_reset_mem = T.inc_subtensor(
            new_mem[output_spikes.nonzero()], -1.)
    # Store refractory
    new_refractory = T.set_subtensor(
        self.refrac_until[output_spikes.nonzero()], time + self.tau_refrac)
    updates.append((self.refrac_until, new_refractory))
    updates.append((self.mem, new_and_reset_mem))
    return output_spikes


def softmax_activation(self, impulse, time, updates):
    """Softmax activation."""
    # Destroy impulse if in refractory period
    masked_imp = T.set_subtensor(
        impulse[(self.refrac_until > time).nonzero()], 0.)
    # Add impulse
    new_mem = self.mem + masked_imp
    # Store spiking
    output_spikes, new_and_reset_mem = theano.ifelse.ifelse(
        T.le(rng.uniform(),
             settings['softmax_clockrate'] * settings['dt'] / 1000),
        trigger_spike(new_mem), skip_spike(new_mem))  # Then and else condition
    # Store refractory. In case of a spike we are resetting all neurons, even
    # the ones that didn't spike. However, in the refractory period we only
    # consider those that spiked. May have to change that...
    new_refractory = T.set_subtensor(
        self.refrac_until[output_spikes.nonzero()], time + self.tau_refrac)
    updates.append((self.refrac_until, new_refractory))
    updates.append((self.mem, new_and_reset_mem))
    return output_spikes


def trigger_spike(new_mem):
    """Trigger spike."""
    activ = T.nnet.softmax(new_mem)
    max_activ = T.max(activ, axis=1, keepdims=True)
    output_spikes = T.eq(activ, max_activ).astype('float32')
    return output_spikes, T.zeros_like(new_mem)


def skip_spike(new_mem):
    """Skip spike."""
    return T.zeros_like(new_mem), new_mem


def reset(self):
    """Reset."""
    if self.inbound_nodes[0].inbound_layers:
        reset(self.inbound_nodes[0].inbound_layers[0])
    self.mem.set_value(floatX(np.zeros(self.output_shape)))
    self.refrac_until.set_value(floatX(np.zeros(self.output_shape)))
    self.spiketrain.set_value(floatX(np.zeros(self.output_shape)))
    if settings['online_normalization']:
        self.spikecounts.set_value(floatX(np.zeros(self.output_shape)))
        if self.layer_type in ["MaxPool2DReLU", "SpikeConv2DReLU"]:
            if settings["maxpool_type"] == "avg_max":
                self.avg_spikerate.set_value(
                    floatX(np.zeros(self.output_shape)))
            elif settings["maxpool_type"] == "fir_max":
                self.fir_spikerate.set_value(
                    floatX(np.zeros(self.output_shape)))
            elif settings["maxpool_type"] == "exp_max":
                self.exp_spikerate.set_value(
                    floatX(np.zeros(self.output_shape)))
        self.max_spikerate.set_value(0.0)
        self.v_thresh.set_value(settings['v_thresh'])


def get_input(self):
    """Get input."""
    if self.inbound_nodes[0].inbound_layers:
        if 'input' in self.inbound_nodes[0].inbound_layers[0].name:
            previous_output = self.input
        else:
            previous_output = \
                self.inbound_nodes[0].inbound_layers[0].get_output()
    else:
        previous_output = K.placeholder(shape=self.input_shape)
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
    init_layer(self, self, v_thresh, tau_refrac, kwargs["layer_type"])
    if 'time_var' in kwargs:
        input_layer = self.inbound_nodes[0].inbound_layers[0]
        input_layer.time_var = kwargs['time_var']
        init_layer(self, input_layer, v_thresh,
                   tau_refrac, kwargs["layer_type"])


def init_layer(self, layer, v_thresh, tau_refrac, layer_type=None):
    """init layer."""
    layer.v_thresh = theano.shared(
        np.asarray(v_thresh, dtype=theano.config.floatX), 'v_thresh')
    layer.tau_refrac = tau_refrac
    layer.refrac_until = shared_zeros(self.output_shape)
    layer.mem = shared_zeros(self.output_shape)
    layer.spiketrain = shared_zeros(self.output_shape)
    layer.updates = []
    layer.layer_type = layer_type_dict[layer_type] \
        if layer_type in layer_type_dict else layer_type
    if settings['online_normalization']:
        layer.spikecounts = shared_zeros(self.output_shape)
        if layer.layer_type in ["MaxPool2DReLU", "SpikeConv2DReLU"]:
            if settings["maxpool_type"] == "avg_max":
                layer.avg_spikerate = shared_zeros(self.output_shape)
            elif settings["maxpool_type"] == "fir_max":
                layer.fir_spikerate = shared_zeros(self.output_shape)
            elif settings["maxpool_type"] == "exp_max":
                layer.exp_spikerate = shared_zeros(self.output_shape)
        layer.max_spikerate = theano.shared(np.asarray(0.0, 'float32'))
    if len(layer.get_weights()) > 0:
        layer.W = K.variable(layer.get_weights()[0])
        layer.b = K.variable(layer.get_weights()[1])


def get_new_thresh(self, time):
    """Get new threshhold."""
    return theano.ifelse.ifelse(
        T.eq(time / settings['dt'] % settings['timestep_fraction'], 0) *
        T.gt(self.max_spikerate, settings['diff_to_min_rate'] / 1000) *
        T.gt(1 / settings['dt'] - self.max_spikerate,
             settings['diff_to_max_rate'] / 1000),
        self.max_spikerate, self.v_thresh)


class SpikeFlatten(Flatten):
    """Spike flatten layer."""

    def __init__(self, label=None, **kwargs):
        """Init function."""
        super().__init__(**kwargs)
        if label is not None:
            self.label = label
        else:
            self.label = self.name

    def get_output(self, train=False):
        """Get output."""
        # Recurse
        inp, time, updates = get_input(self)
        self.updates = updates
        reshaped_inp = T.reshape(inp, self.output_shape)
        return reshaped_inp

    def get_name(self):
        """Get class name."""
        return self.__class__.__name__


class SpikeDense(Dense):
    """Spike Dense layer.

    batch_size x input_shape x out_shape
    """

    def __init__(self, output_dim, weights=None, label=None, **kwargs):
        """Init function."""
        super().__init__(output_dim, weights=weights, **kwargs)
        if label is not None:
            self.label = label
        else:
            self.label = self.name

    def get_output(self, train=False):
        """Get output."""
        # Recurse
        inp, time, updates = get_input(self)

        if settings['online_normalization']:
            # Modify threshold if firing rate of layer too low
            updates.append((self.v_thresh, get_new_thresh(self, time)))

        # Get impulse
        self.impulse = T.add(T.dot(inp, self.W), self.b)
        output_spikes = update_neurons(self, self.impulse, time, updates)
        self.updates = updates
        return T.cast(output_spikes, 'float32')

    def get_name(self):
        """Get class name."""
        return self.__class__.__name__


def pool_same_size(data_in, patch_size, ignore_border=True,
                   st=None, padding=(0, 0)):
    """Max-pooling in same size.

    The indices of maximum values are 1s, else are 0s.

    Parameters
    ----------
    data_in : 4-D tensor.
        input images. Max-pooling will be done over the 2 last dimensions.
    patch_size : tuple
        with length 2 (patch height, patch width)
    ignore_border : bool
        When True, (5,5) input with ds=(2,2) will generate a (2,2) output.
        (3,3) otherwise.
    st : tuple
        Stride size, which is the number of shifts over rows/cols to get the
        next pool region. If st is None, it is considered equal to ds
        (no overlap on pooling regions).
    padding : tuple
        (pad_h, pad_w) pad zeros to extend beyond four borders of the
        images, pad_h is the size of the top and bottom margins, and
        pad_w is the size of the left and right margins.
    """
    output = pool.Pool(ds=patch_size, ignore_border=ignore_border,
                       st=st, padding=padding)(data_in)
    outs = pool.MaxPoolGrad(ds=patch_size, ignore_border=ignore_border,
                            st=st, padding=padding)(data_in, output,
                                                    output) > 0.
    return T.cast(outs, 'float32')


class SpikeConv2DReLU(Convolution2D):
    """Spike 2D Convolution relu.

    batch_size x input_shape x out_shape
    """

    def __init__(self, nb_filter, nb_row, nb_col, weights=None,
                 border_mode='valid', subsample=(1, 1), label=None,
                 filter_flip=True, **kwargs):
        """Init function."""
        super().__init__(nb_filter, nb_row, nb_col, weights=weights,
                         border_mode=border_mode, subsample=subsample,
                         **kwargs)
        self.filter_flip = filter_flip
        if label is not None:
            self.label = label
        else:
            self.label = self.name

    def get_output(self, train=False):
        """Get output."""
        # Recurse
        inp, time, updates = get_input(self)

        if settings['online_normalization']:
            # Modify threshold if firing rate of layer too low
            updates.append((self.v_thresh, get_new_thresh(self, time)))

        # CALCULATE SYNAPTIC SUMMED INPUT
        border_mode = self.border_mode
        if on_gpu() and dnn.dnn_available():
            conv_mode = 'conv' if self.filter_flip else 'cross'
            if border_mode == 'same':
                assert(self.subsample == (1, 1))
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
            conv_out = T.nnet.conv2d(inp, self.W, border_mode=border_mode,
                                     subsample=self.subsample,
                                     filter_flip=self.filter_flip)
            if self.border_mode == 'same':
                shift_x = (self.nb_row - 1) // 2
                shift_y = (self.nb_col - 1) // 2
                conv_out = conv_out[:, :, shift_x:inp.shape[2] + shift_x,
                                    shift_y:inp.shape[3] + shift_y]

        self.impulse = conv_out + K.reshape(self.b, (1, self.nb_filter, 1, 1))

        output_spikes = update_neurons(self, self.impulse, time, updates)
        self.updates = updates
        return T.cast(output_spikes, 'float32')

    def get_name(self):
        """Get class name."""
        return self.__class__.__name__


class AvgPool2DReLU(AveragePooling2D):
    """batch_size x input_shape x out_shape."""

    def __init__(self, pool_size=(2, 2), strides=None, border_mode='valid',
                 label=None, **kwargs):
        """Init average pooling."""
        super().__init__(pool_size=pool_size, strides=strides,
                         border_mode=border_mode)
        if label is not None:
            self.label = label
            self.name = label
        else:
            self.label = self.name

    def get_output(self, train=False):
        """Get Output."""
        # Recurse
        inp, time, updates = get_input(self)

        # CALCULATE SYNAPTIC SUMMED INPUT
        self.impulse = K.pool2d(inp, self.pool_size, self.strides,
                                self.border_mode, pool_mode='avg')

        output_spikes = update_neurons(self, self.impulse, time, updates)
        self.updates = updates
        return T.cast(output_spikes, 'float32')

    def get_name(self):
        """Get class name."""
        return self.__class__.__name__


class MaxPool2DReLU(MaxPooling2D):
    """Max Pooling ReLU.

    batch_size x input_shape x out_shape.
    """

    def __init__(self, pool_size=(2, 2), strides=None, ignore_border=True,
                 label=None, pool_type="fir_max", **kwargs):
        """Init function."""
        self.ignore_border = ignore_border
        super().__init__(pool_size=pool_size, strides=strides)
        if label is not None:
            self.label = label
            self.name = label
        else:
            self.label = self.name
        self.pool_type = pool_type

    def get_output(self, train=False):
        """Get Output.

        Parameters
        ----------
        option : string
            avg_max : max depending on moving average max
            fir_max : max depdnding on first arrived spike.
        """
        # Recurse
        inp, time, updates = get_input(self)

        if self.pool_type == "avg_max":
            spikerate = self.inbound_nodes[0].inbound_layers[0].avg_spikerate \
                        if self.inbound_nodes[0].inbound_layers \
                        else K.placeholder(shape=self.input_shape)
        elif self.pool_type == "fir_max":
            spikerate = self.inbound_nodes[0].inbound_layers[0].fir_spikerate \
                        if self.inbound_nodes[0].inbound_layers \
                        else K.placeholder(shape=self.input_shape)
        elif self.pool_type == "exp_max":
            spikerate = self.inbound_nodes[0].inbound_layers[0].exp_spikerate \
                        if self.inbound_nodes[0].inbound_layers \
                        else K.placeholder(shape=self.input_shape)

        if (self.pool_type in ["avg_max", "fir_max", "exp_max"]) \
           and settings['online_normalization']:
            max_idx = pool_same_size(spikerate,
                                     patch_size=self.pool_size,
                                     ignore_border=self.ignore_border,
                                     st=self.strides)
            self.impulse = pool.pool_2d(inp*max_idx, ds=self.pool_size,
                                        st=self.strides,
                                        ignore_border=self.ignore_border,
                                        mode='max')
        else:
            print("Online Normalization is not enabled, "
                  "or wrong max pooling type, "
                  "choose Average Pooling automatically")
            self.impulse = pool.pool_2d(inp, ds=self.pool_size,
                                        st=self.strides,
                                        ignore_border=self.ignore_border,
                                        mode='average_inc_pad')

        output_spikes = update_neurons(self, self.impulse, time, updates)
        self.updates = updates
        return T.cast(output_spikes, 'float32')

    def get_name(self):
        """Get class name."""
        return self.__class__.__name__


custom_layers = {'SpikeFlatten': SpikeFlatten,
                 'SpikeDense': SpikeDense,
                 'SpikeConv2DReLU': SpikeConv2DReLU,
                 'AvgPool2DReLU': AvgPool2DReLU,
                 'MaxPool2DReLU': MaxPool2DReLU}

layer_type_dict = {'Flatten': 'SpikeFlatten',
                   'Dense': 'SpikeDense',
                   'Convolution2D': 'SpikeConv2DReLU',
                   'AveragePooling2D': 'AvgPool2DReLU',
                   'MaxPooling2D': 'MaxPool2DReLU'}
