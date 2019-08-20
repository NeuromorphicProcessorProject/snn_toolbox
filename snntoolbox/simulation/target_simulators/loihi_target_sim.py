# -*- coding: utf-8 -*-
"""
Building and running spiking neural networks using Intel's Loihi platform.
@author: rbodo
"""

from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import warnings

import numpy as np
from future import standard_library
import keras

import nxsdk_modules.dnn.src.dnn_layers as loihi_snn
from snntoolbox.parsing.utils import get_type
from snntoolbox.conversion.utils import get_scale_fac
from snntoolbox.simulation.utils import AbstractSNN, is_spiking
from snntoolbox.simulation.plotting import plot_probe
from snntoolbox.utils.utils import to_integer

standard_library.install_aliases()


class SNN(AbstractSNN):
    """Class to hold the compiled spiking neural network.

    Represents the compiled spiking neural network, ready for testing in a
    spiking simulator.

    Attributes
    ----------

    """

    def __init__(self, config, queue=None):

        AbstractSNN.__init__(self, config, queue)

        self.snn = None
        self._spiking_layers = {}
        self.spike_probes = None
        self.voltage_probes = None
        self.threshold_scales = None
        partition = self.config.get('loihi', 'partition', fallback='')
        self.partition = None if partition == '' else partition
        self._previous_layer_name = None
        self.do_probe_spikes = \
            any({'spiketrains', 'spikerates', 'correlation', 'spikecounts',
                 'hist_spikerates_activations'} & self._plot_keys) \
            or 'spiketrains_n_b_l_t' in self._log_keys
        self.num_weight_bits = eval(self.config.get(
            'loihi', 'connection_kwargs'))['numWeightBits']

    @property
    def is_parallelizable(self):
        return False

    def get_layer_kwargs(self, layer):

        layer_kwargs = layer.get_config()
        compartment_kwargs = eval(self.config.get('loihi',
                                                  'compartment_kwargs'))
        scale = self.threshold_scales[layer.name]
        compartment_kwargs['vThMant'] = \
            int(compartment_kwargs['vThMant'] * 2 ** scale)
        if self.do_probe_spikes:
            compartment_kwargs['probeSpikes'] = True
        layer_kwargs.update(compartment_kwargs)

        connection_kwargs = eval(self.config.get('loihi', 'connection_kwargs'))
        connection_kwargs['weightExponent'] += \
            self.threshold_scales[self._previous_layer_name]
        layer_kwargs.update(connection_kwargs)

        vp = self.config.getboolean('loihi', 'visualize_partitions',
                                    fallback='')
        if vp != '':
            layer_kwargs['visualizePartitions'] = vp

        encoding = self.config.get('loihi', 'synapse_encoding', fallback='')
        if encoding != '':
            layer_kwargs['synapseEncoding'] = encoding

        return layer_kwargs

    def add_input_layer(self, input_shape):

        if self._poisson_input:
            raise NotImplementedError

        name = self.parsed_model.layers[0].name

        compartment_kwargs = eval(self.config.get('loihi',
                                                  'compartment_kwargs'))
        scale = self.threshold_scales[name]
        compartment_kwargs['vThMant'] = \
            int(compartment_kwargs['vThMant'] * 2 ** scale)
        if self.do_probe_spikes:
            compartment_kwargs['probeSpikes'] = True
        input_layer = loihi_snn.NxInputLayer(input_shape[1:], input_shape[0],
                                             **compartment_kwargs)
        self._spiking_layers[name] = input_layer.input
        self._previous_layer_name = name

    def add_layer(self, layer):

        nx_layer_type = 'Nx' + get_type(layer)

        if not hasattr(loihi_snn, nx_layer_type):
            return

        spike_layer_name = getattr(loihi_snn, nx_layer_type)

        layer_kwargs = self.get_layer_kwargs(layer)

        spike_layer = spike_layer_name(**layer_kwargs)

        inbound = self._spiking_layers[self._previous_layer_name]

        self._spiking_layers[layer.name] = spike_layer(inbound)

        # Convert weights to integers.
        if len(layer.weights):
            weights, biases = layer.get_weights()
            weights, biases = to_integer(weights, biases, self.num_weight_bits)

            if 'Flatten' in self._previous_layer_name:
                pl = self.parsed_model.get_layer(self._previous_layer_name)
                shape = pl.input_shape[1:]
                permutation = np.ravel(np.reshape(
                    np.arange(int(np.prod(shape))), shape, 'F'), 'C')
                weights = weights[permutation]

            spike_layer.set_weights([weights, biases])

        elif 'AveragePooling' in get_type(layer):
            weights, biases = spike_layer.get_weights()
            weights, biases = to_integer(weights, biases, self.num_weight_bits)
            spike_layer.set_weights([weights, biases])

        self._previous_layer_name = layer.name

    def build_dense(self, layer):
        """

        Parameters
        ----------
        layer : keras.layers.Dense

        Returns
        -------

        """

        if layer.activation.__name__ == 'softmax':
            warnings.warn("Activation 'softmax' not implemented. Using 'relu' "
                          "activation instead.", RuntimeWarning)

    def build_convolution(self, layer):
        pass

    def build_pooling(self, layer):
        if layer.__class__.__name__ == 'MaxPooling2D':
            warnings.warn("Layer type 'MaxPooling' not supported yet. " +
                          "Falling back on 'AveragePooling'.", RuntimeWarning)

    def compile(self):

        logdir = self.config.get('paths', 'log_dir_of_current_run')
        input_layer = self._spiking_layers[self.parsed_model.layers[0].name]
        output_layer = self._spiking_layers[self._previous_layer_name]
        self.snn = loihi_snn.NxModel(input_layer, output_layer, verbose=True,
                                     logdir=logdir)

        if self.config.getboolean('loihi', 'save_output', fallback=''):
            path = logdir
        else:
            path = None

        self.snn.compileModel(saveOutputTo=path)

        self.set_vars_to_record()

    def simulate(self, **kwargs):

        data = kwargs[str('x_b_l')]

        self.set_inputs(data)

        lenInterval = 1000
        if self._duration <= lenInterval:
            self.snn.run(self._duration, partition=self.partition)
        else:
            numIntervals = self._duration // lenInterval
            for _ in range(numIntervals):
                self.snn.run(lenInterval, partition=self.partition)

        print("\nCollecting results...")
        output_b_l_t = self.get_recorded_vars(self.snn.layers)

        return output_b_l_t

    def reset(self, sample_idx):

        print("Resetting membrane potentials...")
        for layer in self.snn.layers:
            if not is_spiking(layer, self.config):
                continue
            for i in range(int(np.prod(layer.output_shape[1:]))):
                layer[i].voltage = 0
        print("Done.")

    def end_sim(self):

        self.snn.disconnect()

    def save(self, path, filename):

        pass

    def load(self, path, filename):

        raise NotImplementedError

    def init_cells(self):

        pass

    def set_vars_to_record(self):
        """Set variables to record during simulation."""

        a = loihi_snn.ProbableStates.ACTIVITY
        v = loihi_snn.ProbableStates.VOLTAGE
        s = loihi_snn.ProbableStates.SPIKE

        do_probe_v = \
            'mem_n_b_l_t' in self._log_keys or 'v_mem' in self._plot_keys

        self.spike_probes = {}
        if do_probe_v:
            self.voltage_probes = {}

        for layer in self.snn.layers:
            if not is_spiking(layer, self.config):
                continue

            if self.do_probe_spikes:
                self.spike_probes[layer.name] = []
            if do_probe_v:
                self.voltage_probes[layer.name] = []

            num_neurons = int(np.prod(layer.output_shape[1:]))
            for i in range(num_neurons):
                if self.do_probe_spikes:
                    self.spike_probes[layer.name].append(layer[i].probe(a))
                if do_probe_v:
                    self.voltage_probes[layer.name].append(layer[i].probe(v))

        # The spikes of the last layer are recorded by default because they
        # contain the networks output (classification guess). We can use spike
        # probes here instead of activity traces because the output layer has
        # no shared output axons.
        output_layer = get_spiking_output_layer(self.snn.layers, self.config)
        num_neurons = int(np.prod(output_layer.output_shape[1:]))
        self.spike_probes[output_layer.name] = [output_layer[i].probe(s)
                                                for i in range(num_neurons)]

    def get_spiketrains(self, **kwargs):

        j = self._spiketrains_container_counter
        if self.spiketrains_n_b_l_t is None \
                or j >= len(self.spiketrains_n_b_l_t):
            return

        # Outer for-loop that calls this function starts with
        # 'monitor_index' = 0, but this is reserved for the input and handled
        # by `get_spiketrains_input()`.
        i = kwargs[str('monitor_index')]
        if i == 0:
            return

        layer = self.snn.layers[i]
        if not is_spiking(layer, self.config):
            return

        name = layer.name
        probes = self.stack_layer_probes(self.spike_probes[name])
        shape = self.spiketrains_n_b_l_t[j][0].shape
        spiketrains_b_l_t = self.reshape_flattened_spiketrains(probes, shape)
        # Need to integer divide by max value that soma traces assume, to get
        # rid of the decay tail of the soma trace. The peak value (marking a
        # spike) is defined as 127 in probe creation and will be mapped to 1.
        # (If this is the output layer, we are probing the spikes directly and
        # do not need to scale.)
        is_output_layer = \
            get_spiking_output_layer(self.snn.layers, self.config).name == name
        scale = 1 if is_output_layer else 127
        return spiketrains_b_l_t // scale

    def get_spiketrains_input(self):

        layer = self.snn.layers[0]
        if layer.name not in self.spike_probes:
            return

        probes = self.stack_layer_probes(self.spike_probes[layer.name])
        shape = list(self.parsed_model.input_shape) + [self._num_timesteps]
        spiketrains_b_l_t = self.reshape_flattened_spiketrains(probes, shape)
        return spiketrains_b_l_t // 127

    def get_spiketrains_output(self):

        layer = get_spiking_output_layer(self.snn.layers, self.config)
        probes = self.stack_layer_probes(self.spike_probes[layer.name])
        shape = [self.batch_size, self.num_classes, self._num_timesteps]
        spiketrains_b_l_t = self.reshape_flattened_spiketrains(probes, shape)
        return spiketrains_b_l_t

    def get_vmem(self, **kwargs):
        if self.voltage_probes is None:
            return

        i = kwargs[str('monitor_index')]

        if not is_spiking(self.snn.layers[i], self.config):
            return

        name = self.snn.layers[i].name
        probes = self.voltage_probes[name]
        # Plot instead of returning input layer probes because the toolbox
        # does not expect input to record the membrane potentials.
        if i == 0:
            plot_probe(probes,
                       self.config.get('paths', 'log_dir_of_current_run'),
                       'v_input.png')
        else:
            return self.stack_layer_probes(probes)

    def stack_layer_probes(self, probes):

        return np.stack([p.data[-self._num_timesteps:] for p in probes])

    def reshape_flattened_spiketrains(self, spiketrains, shape, is_list=True):

        # Temporarily move time axis so we can reshape in Fortran style.
        new_shape = shape[:-1]
        new_shape = np.insert(new_shape, 1, shape[-1])

        # Need to flatten in 'C' mode first to stack the timevectors together,
        # then reshape in 'F' style.
        arr = np.reshape(np.ravel(spiketrains), new_shape, 'F')

        # Finally, move the time axis back again.
        return np.moveaxis(arr, 1, -1)

    def set_spiketrain_stats_input(self):
        AbstractSNN.set_spiketrain_stats_input(self)

    def set_inputs(self, inputs):
        inputs = np.ravel(inputs, 'F')
        # Normalize inputs and scale up to 8 bit.
        inputs = (inputs / np.max(inputs) * (2 ** 8 - 1)).astype(int)

        for i, biasMant in enumerate(inputs):
            self.snn.layers[0][i].biasMant = biasMant
            self.snn.layers[0][i].phase = 2

    def preprocessing(self, **kwargs):
        do_process = True
        if do_process:
            print("Normalizing thresholds.")
            self.threshold_scales = normalize_loihi_network(
                self.parsed_model, self.config, **kwargs)
        else:
            print("Skipping threshold normalization.")
            self.threshold_scales = {layer.name: 1
                                     for layer in self.parsed_model.layers}


def get_shape_from_label(label):
    """
    Extract the output shape of a flattened pyNN layer from the layer name
    generated during parsing.

    Parameters
    ----------

    label: str
        Layer name containing shape information after a '_' separator.

    Returns
    -------

    : list
        The layer shape.

    Example
    -------
        >>> get_shape_from_label('02Conv2D_16x32x32')
        [16, 32, 32]

    """
    return [int(i) for i in label.split('_')[1].split('x')]


def normalize_loihi_network(parsed_model, config, **kwargs):

    if 'x_norm' in kwargs:
        x_norm = kwargs[str('x_norm')]  # Values in range [0, 1]
    elif 'x_test' in kwargs:
        x_norm = kwargs[str('x_test')]
    elif 'dataflow' in kwargs:
        x_norm, y = kwargs[str('dataflow')].next()
    else:
        raise NotImplementedError
    print("Using {} samples for normalization.".format(len(x_norm)))
    sizes = [
        len(x_norm) * np.array(layer.output_shape[1:]).prod() * 32 /
        (8 * 1e9) for layer in parsed_model.layers if len(layer.weights) > 0]
    size_str = ['{:.2f}'.format(s) for s in sizes]
    print("INFO: Need {} GB for layer activations.\n".format(size_str))

    batch_size = config.getint('simulation', 'batch_size')

    connection_kwargs = eval(config.get('loihi', 'connection_kwargs'))
    compartment_kwargs = eval(config.get('loihi', 'compartment_kwargs'))
    # Weights have a maximum of 8 bits, used for biases as well.
    num_weight_bits = connection_kwargs['numWeightBits']
    threshold_mant = compartment_kwargs['vThMant']
    weight_exponent = connection_kwargs['weightExponent']
    bias_exponent = compartment_kwargs['biasExp']

    # Loihi limits on weights and thresholds.
    _weight_exponent_lim = 2 ** 3 - 1
    _threshold_mant_lim = 2 ** 17 - 1

    # Loihi applies a fix scaling on weights and thresholds.
    _weight_gain = 2 ** 6
    _threshold_gain = 2 ** 6

    # Input should already be normalized, but do it again just for safety.
    x = x_norm / np.max(x_norm)
    # Convert to integers and scale by bias exponent.
    x *= (2 ** num_weight_bits - 1) * 2 ** bias_exponent

    desired_threshold_to_input_ratio = \
        eval(config.get('loihi', 'desired_threshold_to_input_ratio'))

    scales = {}

    model_copy = keras.models.clone_model(parsed_model)
    model_copy.set_weights(parsed_model.get_weights())

    # Want to optimize thr, while keeping weights and biases in right range.
    for i, layer in enumerate(model_copy.layers):

        print(layer.name)

        prev_scale = scales.get(model_copy.layers[i - 1].name, None)

        if len(layer.weights) > 0:
            # Unconstrained floats
            weights, biases = layer.get_weights()
            # Scale to 8 bit using a common factor for both weights and biases.
            # The weights and biases variables represent mantissa values with
            # zero exponent.
            weights, biases = to_integer(weights, biases, num_weight_bits)
            weights = \
                weights * _weight_gain * 2 ** (weight_exponent + prev_scale)
            check_q_overflow(weights, 1 / desired_threshold_to_input_ratio)
            layer.set_weights([weights, biases * 2 ** bias_exponent])

        # Need to remove softmax in output layer to get activations above 1.
        if hasattr(layer, 'activation') and \
                layer.activation.__name__ == 'softmax':
            layer.activation = keras.activations.relu

        # Get the excitatory post-synaptic potential for each neuron in layer.
        y = keras.models.Sequential([layer]).predict(x, batch_size) if i else x

        if i > 0:
            # Layers like Flatten do not have spiking neurons and therefore no
            # threshold to tune. So we only need to update the input to the
            # next layer, and propagate the scale. The input layer (i == 0)
            # counts as spiking.
            if not is_spiking(layer, config):
                x = y
                scales[layer.name] = prev_scale
                continue
            # Loihi AveragePooling layers get weights of ones. To reproduce
            # this in our Keras model, we need to apply the same
            # transformations as for a regular layer that has weights.
            elif 'AveragePooling' in get_type(layer):
                a = np.prod(layer.pool_size) * (2 ** num_weight_bits - 1)
                y = y * _weight_gain * 2 ** (weight_exponent + prev_scale) * a

        # The highest EPSP determines whether to raise threshold.
        y_max = get_scale_fac(y[np.nonzero(y)], 100)
        print("Maximum increase in compartment voltage per timestep: {}."
              "".format(int(y_max)))

        initial_threshold_to_input_ratio = \
            threshold_mant * _threshold_gain / y_max
        # The gain represents how many powers of 2 the spikerates currently
        # differ from the desired spikerates.
        gain = np.log2(initial_threshold_to_input_ratio /
                       desired_threshold_to_input_ratio)
        print("The ratio of threshold to activations is off by 2**{:.2f}"
              "".format(gain))

        scale = 0
        # Want to find scale exponent for threshold such that
        # -1 < gain + scale < 1. (By using the limits [-1, 1] we allow for a
        # difference of up to a factor 2 in either direction.)
        while gain + scale < -1:
            # First case, gain < 0: Activations in layer are too high, which
            # will result in information loss due to reset-to-zero. Increase
            # threshold.
            if weight_exponent + scale + 1 <= _weight_exponent_lim and \
                    threshold_mant * 2 ** (scale + 1) <= _threshold_mant_lim:
                scale += 1
            else:
                print("Reached upper limit of weight exponent or threshold.")
                break
        else:
            while gain + scale > 1:
                # Second case, gain > 0: Activations in layer are too low,
                # which will result in low spike rates and thus quantization
                # errors. Decrease threshold.
                if weight_exponent + scale - 1 >= - _weight_exponent_lim - 1 \
                        and threshold_mant * 2 ** (scale - 1) >= 1:
                    scale -= 1
                else:
                    print("Reached lower limit of weight exponent or "
                          "threshold.")
                    break

        print("Scaling thresholds of layer {} and weights of subsequent layer "
              "by 2**{}.\n".format(layer.name, scale))
        scales[layer.name] = scale

        # Apply activation function (dividing by threshold) to obtain the
        # output of the current layer, which will be used as input to the next.
        beta = threshold_mant * _threshold_gain * 2 ** scale
        x = y / beta
        # Apply the same scaling to the weights of current layer, which does
        # not affect the subsequent iterations in this loop, but which scales
        # the activations of the parsed layer later when plotting against the
        # spike rates.
        if len(layer.weights) > 0:
            weights, biases = layer.get_weights()
            layer.set_weights([weights / beta, biases / beta])

    parsed_model.set_weights(model_copy.get_weights())

    return scales


def check_q_overflow(weights, p):
    num_channels = weights.shape[-1]
    weights_flat = np.reshape(weights, (-1, num_channels))
    q_min = - 2 ** 15
    q_max = - q_min - 1
    weighted_fanin = np.sum(weights_flat, 0)
    neg = np.mean(weighted_fanin < q_min)
    pos = np.mean(weighted_fanin > q_max)
    if neg or pos:
        print("In the worst case of all pre-synaptic neurons firing "
              "simultaneously, the dendritic accumulator will overflow in "
              "{:.2%} and underflow in {:.2%} of neurons.".format(pos, neg))
        print("Estimating averages...")
        neg = []
        pos = []
        num_fanin = len(weights_flat)
        for i in range(2 ** min(num_fanin, 16)):
            spikes = np.random.binomial(1, p, num_fanin)
            weighted_fanin = np.sum(weights_flat[spikes > 0], 0)
            neg.append(np.mean(weighted_fanin < q_min) * 100)
            pos.append(np.mean(weighted_fanin > q_max) * 100)
        print("On average, the dendritic accumulator will overflow in {:.2f} "
              "+/- {:.2f} % and underflow in {:.2f} +/- {:.2f} % of neurons."
              "".format(np.mean(pos), np.std(pos), np.mean(neg), np.std(neg)))


def overflow_signed(x, num_bits):
    """Compute overflow on an array of signed integers.

    Parameters
    ----------
    x : ndarray
        Integer values for which to compute values after overflow.
    num_bits : int
        Number of bits, not including sign, to compute overflow for.

    Returns
    -------
    out : ndarray
        Values of x overflowed as would happen with limited bit representation.
    overflowed : ndarray
        Boolean array indicating which values of ``x`` actually overflowed.
    """

    x = x.astype(int)

    lim = 2 ** num_bits
    smask = np.array(lim, int)  # mask for the sign bit
    xmask = smask - 1  # mask for all bits <= `bits`

    # Find where x overflowed
    overflowed = (x < -lim) | (x >= lim)

    zmask = x & smask  # if `out` has negative sign bit, == 2**bits
    out = x & xmask  # mask out all bits > `bits`
    out -= zmask  # subtract 2**bits if negative sign bit

    return out, overflowed


def to_mantexp(x, mant_max, exp_max):
    r = np.maximum(np.abs(x) / mant_max, 1)
    exp = np.ceil(np.log2(r)).astype(int)
    assert np.all(exp <= exp_max)
    man = np.round(x / 2 ** exp).astype(int)
    assert np.all(np.abs(man) <= mant_max)
    return man, exp


def get_spiking_output_layer(layers, config):
    for layer in reversed(layers):
        if is_spiking(layer, config):
            return layer
