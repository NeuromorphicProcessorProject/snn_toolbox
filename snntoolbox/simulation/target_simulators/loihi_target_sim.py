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

from scratch.cnn_mode.cnn_mode import LoihiInputLayer, LoihiModel, \
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
        compartment_kwargs['vThMant'] *= 2 ** scale
        input_layer = LoihiInputLayer(input_shape[1:], input_shape[0],
                                      **compartment_kwargs)
        self._spiking_layers[name] = input_layer.input

    def add_layer(self, layer):

        layer_name = layer.name

        if 'Flatten' in layer_name:
            self.flatten_shapes.append(get_shape_from_label(
                self._previous_layer_name))
            return

        from snntoolbox.parsing.utils import get_type
        import scratch.cnn_mode.cnn_mode as loihi_snn
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
        layer_kwargs.update(compartment_kwargs)

        connection_kwargs = eval(self.config.get('loihi', 'connection_kwargs'))
        if self._previous_layer_name is not None:
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

        self.snn.run(self._duration, partition=self.partition)

        print("\nCollecting results...")
        output_b_l_t = self.get_recorded_vars(self.snn.layers)

        return output_b_l_t

    def reset(self, sample_idx):

        print("Resetting membrane potentials...")
        for layer in self.snn.layers:
            for i in range(len(layer)):
                layer[i].voltage = 0
                layer[i].phase = 2
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

        u = ProbableStates.CURRENT
        v = ProbableStates.VOLTAGE

        do_probe_u = \
            any({'spiketrains', 'spikerates', 'correlation', 'spikecounts',
                 'hist_spikerates_activations'} & self._plot_keys) \
            or 'spiketrains_n_b_l_t' in self._log_keys

        do_probe_v = \
            'mem_n_b_l_t' in self._log_keys or 'v_mem' in self._plot_keys

        if do_probe_u:
            self.spike_probes = {}
        if do_probe_v:
            self.voltage_probes = {}

        for layer in self.snn.layers:
            if do_probe_u:
                self.spike_probes[layer.name] = []
            if do_probe_v:
                self.voltage_probes[layer.name] = []

            num_neurons = int(np.prod(layer.output_shape[1:]))
            for i in range(num_neurons):
                if do_probe_u:
                    self.spike_probes[layer.name].append(layer[i].probe(u))
                if do_probe_v:
                    self.voltage_probes[layer.name].append(layer[i].probe(v))

        # The spikes of the last layer are recorded by default because they
        # contain the networks output (classification guess).
        if not do_probe_u:
            output_layer = self.snn.layers[-1]
            num_neurons = int(np.prod(output_layer.output_shape[1:]))
            self.spike_probes = {output_layer.name:
                                 [output_layer[i].probe(u)
                                  for i in range(num_neurons)]}

    def get_spiketrains(self, **kwargs):
        if self.spike_probes is None:
            return

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
        return spiketrains_b_l_t / layer.compartmentKwargs['vThMant'] / 2 ** 6

    def get_spiketrains_input(self):
        if self.spike_probes is None:
            return

        layer = self.snn.layers[-1]
        probes = self.stack_layer_probes(self.spike_probes[layer.name])
        shape = list(self.parsed_model.input_shape) + [self._num_timesteps]
        spiketrains_b_l_t = self.reshape_flattened_spiketrains(probes, shape)
        return spiketrains_b_l_t / layer.compartmentKwargs['vThMant'] / 2 ** 6

    def get_spiketrains_output(self):
        if self.spike_probes is None:
            return

        layer = self.snn.layers[-1]
        probes = self.stack_layer_probes(self.spike_probes[layer.name])
        shape = [self.batch_size, self.num_classes, self._num_timesteps]
        spiketrains_b_l_t = self.reshape_flattened_spiketrains(probes, shape)
        return spiketrains_b_l_t / layer.compartmentKwargs['vThMant'] / 2 ** 6

    def get_vmem(self, **kwargs):
        if self.voltage_probes is None:
            return

        i = kwargs[str('monitor_index')]
        name = self.snn.layers[i].name
        probes = self.stack_layer_probes(self.voltage_probes[name])
        # Plot instead of returning input layer probes because the toolbox
        # does not expect input to record the membrane potentials.
        if i == 0:
            plot_probe(probes,
                       self.config.get('paths', 'log_dir_of_current_run'),
                       'v_input.png')
        else:
            return probes

    def stack_layer_probes(self, probes):

        return np.stack([p.data[-self._num_timesteps:] for p in probes])

    def reshape_flattened_spiketrains(self, spiketrains, shape, is_list=True):

        # Temporarily move time axis so we can reshape in Fortran style.
        new_shape = np.copy(shape)
        new_shape[-1] = shape[1]
        new_shape[1] = shape[-1]

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
        inputs = (inputs / np.max(inputs) * 2 ** 8).astype(int)

        for i, biasMant in enumerate(inputs):
            self.snn.layers[0][i].biasMant = biasMant
            self.snn.layers[0][i].phase = 2

    def preprocessing(self, **kwargs):
        print("Normalizing thresholds.")
        from snntoolbox.conversion.utils import normalize_loihi_network
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
