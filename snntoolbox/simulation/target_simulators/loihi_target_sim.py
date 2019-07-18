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

from nxsdk_modules.dnn.src.dnn_mode import LoihiInputLayer, LoihiModel, \
    ProbableStates
from snntoolbox.simulation.utils import AbstractSNN
from snntoolbox.utils.utils import to_integer
from snntoolbox.simulation.plotting import plot_probe

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

    @property
    def is_parallelizable(self):
        return False

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
        input_layer = LoihiInputLayer(input_shape[1:], input_shape[0],
                                      **compartment_kwargs)
        self._spiking_layers[name] = input_layer.input
        self._previous_layer_name = name

    def add_layer(self, layer):

        layer_name = layer.name

        if 'Flatten' in layer_name:
            self.flatten_shapes.append(get_shape_from_label(
                self._previous_layer_name))
            return

        from snntoolbox.parsing.utils import get_type
        import nxsdk_modules.dnn.src.dnn_mode as loihi_snn
        spike_layer_name = getattr(loihi_snn, 'Loihi' + get_type(layer))
        # noinspection PyProtectedMember
        inbound = [self._spiking_layers[inb.name] for inb in
                   layer._inbound_nodes[0].inbound_layers]
        if len(inbound) == 1:
            inbound = inbound[0]

        layer_kwargs = layer.get_config()
        compartment_kwargs = eval(self.config.get('loihi',
                                                  'compartment_kwargs'))
        scale = self.threshold_scales[layer_name]
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
        layer_kwargs['visualizePartitions'] = None if vp == '' else vp
        encoding = self.config.get('loihi', 'synapse_encoding', fallback='')
        if encoding != '':
            layer_kwargs['synapseEncoding'] = encoding

        spike_layer = spike_layer_name(**layer_kwargs)

        if 'Pool' in layer_name:
            weights, biases = spike_layer.get_weights()
            weights = np.ones_like(weights) / np.prod(spike_layer.pool_size)
            biases = np.zeros_like(biases)
        else:
            weights, biases = layer.get_weights()

        num_weight_bits = eval(self.config.get(
            'loihi', 'connection_kwargs'))['numWeightBits']
        weights, biases = to_integer(weights, biases, num_weight_bits)

        self._spiking_layers[layer_name] = spike_layer(inbound)
        spike_layer.set_weights([weights, biases])
        self._previous_layer_name = layer_name

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

        if len(self.flatten_shapes):
            _layer = self._spiking_layers[self._previous_layer_name]
            weights, biases = _layer.get_weights()
            weights = fix_flatten(weights, self.flatten_shapes.pop(),
                                  self.data_format)
            _layer.set_weights([weights, biases])

    def build_convolution(self, layer):
        pass

    def build_pooling(self, layer):
        if layer.__class__.__name__ == 'MaxPooling2D':
            warnings.warn("Layer type 'MaxPooling' not supported yet. " +
                          "Falling back on 'AveragePooling'.", RuntimeWarning)

    def compile(self):

        input_layer = self._spiking_layers[self.parsed_model.layers[0].name]
        output_layer = self._spiking_layers[self._previous_layer_name]
        self.snn = LoihiModel(input_layer, output_layer)

        self.snn.compileModel()

        self.set_vars_to_record()

    def simulate(self, **kwargs):

        data = kwargs[str('x_b_l')]

        self.set_inputs(data)

        lenInterval = 1000
        if self._duration < lenInterval:
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

        a = ProbableStates.ACTIVITY
        v = ProbableStates.VOLTAGE
        s = ProbableStates.SPIKE

        do_probe_v = \
            'mem_n_b_l_t' in self._log_keys or 'v_mem' in self._plot_keys

        self.spike_probes = {}
        if do_probe_v:
            self.voltage_probes = {}

        for layer in self.snn.layers:
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
        output_layer = self.snn.layers[-1]
        num_neurons = int(np.prod(output_layer.output_shape[1:]))
        self.spike_probes[output_layer.name] = [output_layer[i].probe(s)
                                                for i in range(num_neurons)]

    def get_spiketrains(self, **kwargs):

        j = self._spiketrains_container_counter
        if self.spiketrains_n_b_l_t is None \
                or j >= len(self.spiketrains_n_b_l_t):
            return None

        # Outer for-loop that calls this function starts with
        # 'monitor_index' = 0, but this is reserved for the input and handled
        # by `get_spiketrains_input()`.
        i = len(self.snn.layers) - 1 if kwargs[str('monitor_index')] == -1 \
            else kwargs[str('monitor_index')] + 1
        layer = self.snn.layers[i]
        probes = self.stack_layer_probes(self.spike_probes[layer.name])
        shape = self.spiketrains_n_b_l_t[j][0].shape
        spiketrains_b_l_t = self.reshape_flattened_spiketrains(probes, shape)
        # Need to integer divide by max value that soma traces assume, to get
        # rid of the decay tail of the soma trace. The peak value (marking a
        # spike) is defined as 127 in probe creation and will be mapped to 1.
        # (If this is the output layer, we are probing the spikes directly and
        # do not need to scale.)
        scale = 1 if i == len(self.snn.layers) - 1 else 127
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

        layer = self.snn.layers[-1]
        probes = self.stack_layer_probes(self.spike_probes[layer.name])
        shape = [self.batch_size, self.num_classes, self._num_timesteps]
        spiketrains_b_l_t = self.reshape_flattened_spiketrains(probes, shape)
        return spiketrains_b_l_t

    def get_vmem(self, **kwargs):
        if self.voltage_probes is None:
            return

        i = kwargs[str('monitor_index')]
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
        print("Normalizing thresholds.")
        self.threshold_scales = normalize_loihi_network(self.parsed_model,
                                                        self.config, **kwargs)


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


def fix_flatten(weights, layer_shape, data_format):
    print("Swapping data_format of Flatten layer.")
    if data_format == 'channels_last':
        y_in, x_in, f_in = layer_shape
    else:
        f_in, y_in, x_in = layer_shape
    i_new = []
    for i in range(len(weights)):  # Loop over input neurons
        # Sweep across channel axis of feature map. Assumes that each
        # consecutive input neuron lies in a different channel. This is
        # the case for channels_last, but not for channels_first.
        f = i % f_in
        # Sweep across height of feature map. Increase y by one if all
        # rows along the channel axis were seen.
        y = i // (f_in * x_in)
        # Sweep across width of feature map.
        x = (i // f_in) % x_in
        i_new.append(f * x_in * y_in + y * x_in + x)

    return weights[np.argsort(i_new)]  # Move rows to new i's.


def normalize_loihi_network(parsed_model, config, **kwargs):
    import keras
    from snntoolbox.utils.utils import to_integer
    from snntoolbox.simulation.utils import is_spiking
    from snntoolbox.conversion.utils import get_scale_fac

    x_norm = kwargs[str('x_test')]  # Values in range [0, 1]

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
        if len(layer.weights) > 0:
            scale = scales[model_copy.layers[i - 1].name]
            # Unconstrained floats
            weights, biases = layer.get_weights()
            # Scale to 8 bit using a common factor for both weights and biases.
            # The weights and biases variables represent mantissa values with
            # zero exponent.
            weights, biases = to_integer(weights, biases, num_weight_bits)
            weights = weights * _weight_gain * 2 ** (weight_exponent + scale)
            check_q_overflow(weights, 1 / desired_threshold_to_input_ratio)
            layer.set_weights([weights, biases * 2 ** bias_exponent])

        # Need to remove softmax in output layer to get activations above 1.
        if hasattr(layer, 'activation') and \
                layer.activation.__name__ == 'softmax':
            layer.activation = keras.activations.relu

        # Get the excitatory post-synaptic potential for each neuron in layer.
        y = keras.models.Sequential([layer]).predict(x, batch_size) if i else x

        # Layers like Flatten do not have spiking neurons and therefore no
        # threshold to tune. So we only need to update the input to the next
        # layer, and propagate the scale. The input layer (i == 0) counts as
        # spiking.
        if i > 0 and not is_spiking(layer, config):
            x = y
            scales[layer.name] = scales[model_copy.layers[i - 1].name]
            continue

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
              "by 2**{}.".format(layer.name, scale))
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
