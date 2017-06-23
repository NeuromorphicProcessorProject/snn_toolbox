# -*- coding: utf-8 -*-
"""Building and simulating spiking neural networks using Brian2.

@author: rbodo
"""

from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import warnings

import numpy as np
from future import standard_library

from snntoolbox.simulation.utils import AbstractSNN

standard_library.install_aliases()


class SNN(AbstractSNN):
    """
    Represents the compiled spiking neural network, ready for testing in a
    spiking simulator.

    Attributes
    ----------

    layers: list[brian2.NeuronGroup]
        Each entry represents a layer, i.e. a population of neurons, in form of
        Brian2 ``NeuronGroup`` objects.

    connections: list[brian2.Synapses]
        Brian2 ``Synapses`` objects representing the connections between
        individual layers.

    threshold: str
        Defines spiking threshold.

    v_reset: str
        Defines reset potential.

    eqs: str
        Differential equation for membrane potential.

    spikemonitors: list[brian2.SpikeMonitor]
        Brian2 ``SpikeMonitor`` s for each layer that records spikes.

    statemonitors: list[brian2.StateMonitor]
        Brian2 ``StateMonitor`` s for each layer that records membrane
        potential.

    snn: brian2.Network
        The spiking network.
    """

    def __init__(self, config, queue=None):

        AbstractSNN.__init__(self, config, queue)

        self.layers = []
        self._conns = []  # Temporary container for layer connections.
        self._biases = []  # Temporary container for layer biases.
        self.connections = []  # Final container for all layers.
        self.threshold = 'v >= v_thresh'
        self.v_reset = 'v = v_reset'
        self.eqs = 'v = 0 : 1'
        self.spikemonitors = []
        self.statemonitors = []
        self.snn = None
        self._input_layer = None
        self._cell_params = None

        # Track the output layer spikes. Add monitor here if it was not already
        # appended above (because settings['verbose'] < 1)
        if len(self.spikemonitors) < len(self.layers):
            self.spikemonitors.append(self.sim.SpikeMonitor(self.layers[-1]))

    @property
    def is_parallelizable(self):
        return False

    def add_input_layer(self, input_shape):

        self.layers.append(self.sim.PoissonGroup(
            np.prod(input_shape[1:]), rates=0*self.sim.Hz,
            dt=self._dt*self.sim.ms))
        self.layers[0].add_attribute('label')
        self.layers[0].label = 'InputLayer'
        self.spikemonitors.append(self.sim.SpikeMonitor(self.layers[0]))
        # Need placeholders "None" for layers without states:
        self.statemonitors.append(self.sim.StateMonitor(self.layers[0], [],
                                                        False))

    def add_layer(self, layer):

        if 'Flatten' in layer.__class__.__name__:
            return

        self._conns = []
        self.layers.append(self.sim.NeuronGroup(
            np.prod(layer.output_shape[1:]), self.eqs, 'linear', self.threshold,
            self.v_reset, dt=self._dt*self.sim.ms))
        self.connections.append(self.sim.Synapses(
            self.layers[-2], self.layers[-1], 'w:1', on_pre='v+=w',
            dt=self._dt*self.sim.ms))
        self.layers[-1].add_attribute('label')
        self.layers[-1].label = layer.name
        if 'spiketrains' in self._plot_keys:
            self.spikemonitors.append(self.sim.SpikeMonitor(self.layers[-1]))
        if 'v_mem' in self._plot_keys:
            self.statemonitors.append(self.sim.StateMonitor(self.layers[-1],
                                                            'v', True))

    def build_dense(self, layer):

        if layer.activation == 'softmax':
            raise warnings.warn("Activation 'softmax' not implemented. Using "
                                "'relu' activation instead.", RuntimeWarning)

        weights, self._biases = layer.get_weights()
        self.set_biases()
        self.connections[-1].connect(True)
        self.connections[-1].w = weights.flatten()

    def build_convolution(self, layer):
        from snntoolbox.simulation.utils import build_convolution

        delay = self.config.getfloat('cell', 'delay')
        self._conns, self._biases = build_convolution(layer, delay)

        self.set_biases()

        print("Connecting layer...")
        for conn in self._conns:
            i = conn[0]
            j = conn[1]
            self.connections[-1].connect(i=i, j=j)
            self.connections[-1].w[i, j] = conn[2]

    def build_pooling(self, layer):
        from snntoolbox.simulation.utils import build_pooling

        delay = self.config.getfloat('cell', 'delay')
        self._conns = build_pooling(layer, delay)

        for conn in self._conns:
            self.connections[-1].connect(i=conn[0], j=conn[1])
            self.connections[-1].w = 1 / np.prod(layer.pool_size)

    def compile(self):

        self.snn = self.sim.Network(self.layers, self.connections,
                                    self.spikemonitors, self.statemonitors)
        self.snn.store()

        # Set input layer
        for obj in self.snn.objects:
            if 'poissongroup' in obj.name and 'thresholder' not in obj.name:
                self._input_layer = obj
        assert self._input_layer, "No input layer found."

    def simulate(self, **kwargs):

        if self._poisson_input:
            self._input_layer.rates = kwargs['x_b_l'].flatten() * 1000 / \
                                      self.rescale_fac * self.sim.Hz
        elif self._dataset_format == 'aedat':
            # TODO: Implement by using brian2.SpikeGeneratorGroup.
            raise NotImplementedError
        else:
            try:
                # TODO: Implement constant input by using brian2.TimedArray.
                self._input_layer.current = kwargs['x_b_l'].flatten()
            except AttributeError:
                raise NotImplementedError

        self.snn.run(self._duration*self.sim.ms, namespace=self._cell_params,
                     report='stdout', report_period=10*self.sim.ms)

        output_b_l_t = self.get_recorded_vars(self.layers)

        return output_b_l_t

    def reset(self, sample_idx):
        mod = self.config.getint('simulation', 'reset_between_nth_sample')
        mod = mod if mod else sample_idx + 1
        if sample_idx % mod == 0:
            print("Resetting simulator...")
            self.snn.restore()

    def end_sim(self):

        pass

    def save(self, path, filename):

        warnings.warn("Saving Brian2 spiking model to disk is not yet "
                      "implemented.", RuntimeWarning)

    def load(self, path, filename):

        # TODO: Implement saving and loading Brian2 models.
        raise NotImplementedError("Loading Brian2 spiking model from disk is "
                                  "not yet implemented.")

    def init_cells(self):
        cell_conf = self.config['cell']
        self._cell_params = {
            'v_thresh': cell_conf.getfloat('v_thresh'),
            'v_reset': cell_conf.getfloat('v_reset'),
            'tau_m': cell_conf.getfloat('tau_m') * self.sim.ms}

    def get_spiketrains(self, **kwargs):
        j = self._spiketrains_container_counter
        if j >= len(self.spiketrains_n_b_l_t):
            return None

        shape = self.spiketrains_n_b_l_t[j][0].shape

        # Outer for-loop that calls this function starts with
        # 'monitor_index' = 0, but this is reserved for the input and handled by
        # `get_spiketrains_input()`.
        i = len(self.spikemonitors) - 1 if kwargs['monitor_index'] == -1 else \
            kwargs['monitor_index'] + 1
        spiketrain_dict = self.spikemonitors[i].spike_trains()
        spiketrains_flat = np.array([spiketrain_dict[key] / self.sim.ms for key
                                     in spiketrain_dict.keys()])
        spiketrains_b_l_t = self.reshape_flattened_spiketrains(spiketrains_flat,
                                                               shape)
        return spiketrains_b_l_t

    def get_spiketrains_input(self):
        shape = list(self.parsed_model.input_shape) + [self._num_timesteps]
        spiketrain_dict = self.spikemonitors[0].spike_trains()
        spiketrains_flat = np.array([spiketrain_dict[key] / self.sim.ms for key
                                     in spiketrain_dict.keys()])
        spiketrains_b_l_t = self.reshape_flattened_spiketrains(spiketrains_flat,
                                                               shape)
        return spiketrains_b_l_t

    def get_vmem(self, **kwargs):
        try:
            return np.array([np.array(v).transpose() for v in
                             self.statemonitors[kwargs['monitor_index']].v])
        except AttributeError:
            return None

    def set_biases(self):
        """Set biases.

        Notes
        -----

        This has not been tested yet.
        """

        if any(self._biases):  # TODO: Implement biases.
            warnings.warn("Biases not implemented.", RuntimeWarning)

    def set_spiketrain_stats_input(self):
        AbstractSNN.set_spiketrain_stats_input(self)
