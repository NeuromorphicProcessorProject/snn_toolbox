# -*- coding: utf-8 -*-
"""Building SNNs using Brian2.

The modules in ``target_simulators`` package allow building a spiking network
and exporting it for use in a spiking simulator.

This particular module offers functionality for Brian2 simulator. Adding
another simulator requires implementing the class ``AbstractSNN`` with its
methods tailored to the specific simulator.

Created on Thu May 19 15:00:02 2016

@author: rbodo
"""

from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import warnings
import numpy as np
from future import standard_library
from snntoolbox.target_simulators.common import AbstractSNN

standard_library.install_aliases()


class SNN(AbstractSNN):
    """
    Class to hold the compiled spiking neural network, ready for testing in a
    spiking simulator.

    Attributes
    ----------

    layers: list
        Each entry represents a layer, i.e. a population of neurons, in form of
        Brian2 ``NeuronGroup`` objects.

    connections: list
        Brian2 ``Synapses`` objects representing the connections between
        individual layers.

    threshold: string
        Defines spiking threshold.

    reset: string
        Defines reset potential.

    eqs: string
        Differential equation for membrane potential.

    spikemonitors: list
        Brian2 ``SpikeMonitor`` s for each layer that records spikes.

    statemonitors: list
        Brian2 ``StateMonitor`` s for each layer that records membrane
        potential.

    Methods
    -------

    build:
        Convert an ANN to a spiking neural network, using layers derived from
        Keras base classes.
    run:
        Simulate a spiking network.
    save:
        Write model architecture and parameters to disk.
    load:
        Load model architecture and parameters from disk.
    end_sim:
        Clean up after simulation.
    """

    def __init__(self, config, queue=None):

        AbstractSNN.__init__(self, config, queue)

        self.layers = []
        self._conns = []  # Temporary container for layer connections.
        self._biases = []  # Temporary container for layer biases.
        self.connections = []  # Final container for all layers.
        self.threshold = 'v > v_thresh'
        self.reset = 'v = v_reset'
        self.eqs = 'dv/dt = -v/tau_m : volt'
        self.spikemonitors = []
        self.statemonitors = []
        self.snn = None
        self._input_layer = None
        self._cell_params = None

        # Track the output layer spikes. Add monitor here if it was not already
        # appended above (because settings['verbose'] < 1)
        if len(self.spikemonitors) < len(self.layers):
            self.spikemonitors.append(self.sim.SpikeMonitor(self.layers[-1]))

    def add_input_layer(self, input_shape):

        self.layers.append(self.sim.PoissonGroup(
            np.prod(input_shape[1:]), rates=0*self.sim.Hz,
            dt=self._dt*self.sim.ms))
        self.spikemonitors.append(self.sim.SpikeMonitor(self.layers[0]))
        self.layers[0].add_attribute('label')
        self.layers[0].label = 'InputLayer'

    def add_layer(self, layer):

        if 'Flatten' in layer.__class__.__name__:
            return

        self._conns = []
        self.layers.append(self.sim.NeuronGroup(
            np.prod(layer.output_shape[1:]), model=self.eqs,
            threshold=self.threshold, reset=self.reset, dt=self._dt*self.sim.ms,
            method='linear'))
        self.connections.append(self.sim.Synapses(
            self.layers[-2], self.layers[-1], model='w:volt', on_pre='v+=w',
            dt=self._dt*self.sim.ms))
        self.layers[-1].add_attribute('label')
        self.layers[-1].label = layer.name
        if 'spiketrains' in self._plot_keys:
            self.spikemonitors.append(self.sim.SpikeMonitor(self.layers[-1]))
        if 'v_mem' in self._plot_keys:
            self.statemonitors.append(self.sim.StateMonitor(self.layers[-1],
                                                            'v', record=True))

    def build_dense(self, layer):

        if layer.activation == 'softmax':
            raise warnings.warn("Activation 'softmax' not implemented. Using "
                                "'relu' activation instead.", RuntimeWarning)

        weights, self._biases = layer.get_weights()
        self.set_biases()
        self.connections[-1].connect(True)
        self.connections[-1].w = weights.flatten() * self.sim.volt

    def build_convolution(self, layer):
        from snntoolbox.target_simulators.common import build_convolution

        delay = self.config.getfloat('cell', 'delay')
        self._conns, self._biases = build_convolution(layer, delay)

        self.set_biases()

        print("Connecting layer...")
        for conn in self._conns:
            i = conn[0]
            j = conn[1]
            self.connections[-1].connect(i=i, j=j)
            self.connections[-1].w[i, j] = conn[2] * self.sim.volt
        print("Done.")

    def build_pooling(self, layer):
        from snntoolbox.target_simulators.common import build_pooling

        delay = self.config.getfloat('cell', 'delay')
        self._conns = build_pooling(layer, delay)

        for conn in self._conns:
            self.connections[-1].connect(i=conn[0], j=conn[1])
            self.connections[-1].w = self.sim.volt / np.prod(layer.pool_size)

    def compile(self):

        self.snn = self.sim.Network(self.layers, self.connections,
                                    self.spikemonitors, self.statemonitors)
        # Set input layer
        for obj in self.snn.objects:
            if 'poissongroup' in obj.name and 'thresholder' not in obj.name:
                self._input_layer = obj
        assert self._input_layer, "No input layer found."

    def simulate(self, **kwargs):

        from snntoolbox.io_utils.plotting import plot_potential
        from snntoolbox.core.util import get_layer_ops

        if self._poisson_input:
            self._input_layer.rates = kwargs['x_b_l'].flatten() * 1000 / \
                                      self.rescale_fac * self.sim.Hz
        elif self._dataset_format == 'aedat':
            raise NotImplementedError
        else:
            try:
                # TODO: Implement constant input currents.
                self._input_layer.current = kwargs['x_b_l'].flatten()
            except AttributeError:
                raise NotImplementedError

        self.snn.store()
        self.snn.run(self._duration * self.sim.ms, namespace=self._cell_params)

        out_spikes = self.spikemonitors[-1].spiketrains

        # For each time step, get number of spikes of all neurons in the output
        # layer.
        output_b_l_t = np.zeros((self.batch_size, self.num_classes,
                                 self._num_timesteps), 'int32')
        for k, spiketrain in enumerate(out_spikes):
            for t in range(len(self._num_timesteps)):
                output_b_l_t[0, k, t] = np.count_nonzero(spiketrain <=
                                                         t * self._dt)

        # Record neuron variables.
        for i in range(len(self.layers[1:])):

            # Get spike trains.
            try:
                spiketrain_dict = self.spikemonitors[i].spike_trains()
                spiketrains = np.array([spiketrain_dict[key] / self.sim.ms
                                        for key in spiketrain_dict.keys()])
            except AttributeError:
                continue

            # Convert list of spike times into array where nonzero entries
            # (indicating spike times) are properly spread out across array.
            layer_shape = self.spiketrains_n_b_l_t[i][0].shape
            spiketrains_flat = np.zeros((np.prod(layer_shape),
                                         self._num_timesteps))
            for k, spiketrain in enumerate(spiketrains):
                for t in spiketrain:
                    spiketrains_flat[k, int(t / self._dt)] = t

            # Reshape flat spike train array to original layer shape.
            spiketrains_b_l_t = np.reshape(spiketrains_flat, layer_shape)

            # Add spike trains to log variables.
            if self.spiketrains_n_b_l_t is not None:
                self.spiketrains_n_b_l_t[i][0] = spiketrains_b_l_t

            # Use spike trains to compute the number of operations.
            if self.operations_b_t is not None:
                for t in range(len(self._num_timesteps)):
                    self.operations_b_t[:, t] += get_layer_ops(
                        spiketrains_b_l_t[t], self.fanout[i + 1],
                        self.num_neurons_with_bias[i + 1])

        for i in range(len(self.layers[1:])):

            # Get membrane potentials.
            try:
                mem = np.array([
                    np.true_divide(v, 1e6 / self.sim.mV).transpose() for v in
                    self.statemonitors[i - 1].v])
            except AttributeError:
                continue

            # Reshape flat array to original layer shape.
            layer_shape = self.mem_n_b_l_t[i][0].shape
            self.mem_n_b_l_t[i][0] = np.reshape(mem, layer_shape)

            # Plot membrane potentials of layer.
            times = self.statemonitors[0].t / self.sim.ms
            show_legend = True if i >= len(self.layers) - 2 else False
            plot_potential(times, self.mem_n_b_l_t[i], self.config, show_legend,
                           self.config['paths']['log_dir_of_current_run'])

        # Get spike trains of input layer.
        spiketrain_dict = self.spikemonitors[0].spike_trains()
        spiketrains = np.array([spiketrain_dict[key] / self.sim.ms
                                for key in spiketrain_dict.keys()])

        # Convert list of spike times into array where nonzero entries
        # (indicating spike times) are properly spread out across array.
        layer_shape = self.parsed_model.get_batch_input_shape()
        spiketrains_flat = np.zeros((np.prod(layer_shape), self._num_timesteps))
        for k, spiketrain in enumerate(spiketrains):
            for t in spiketrain:
                spiketrains_flat[k, int(t / self._dt)] = t

        # Reshape flat spike train array to original layer shape.
        input_b_l_t = np.reshape(spiketrains_flat, layer_shape)

        if 'input_b_l_t' in self._log_keys:
            self.input_b_l_t = input_b_l_t

        if self.operations_b_t is not None:
            for t in range(self._duration):
                if self._poisson_input or self._dataset_format == 'aedat':
                    input_ops = get_layer_ops(input_b_l_t[t], self.fanout[0])
                else:
                    input_ops = np.ones(self.batch_size) * self.num_neurons[1]
                    if t == 0:
                        input_ops *= 2 * self.fanin[1]  # MACs for convol.
                self.operations_b_t[:, t] += input_ops

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
            'v_thresh': cell_conf.getfloat('v_thresh') * self.sim.volt,
            'v_reset': cell_conf.getfloat('v_reset') * self.sim.volt,
            'tau_m': cell_conf.getfloat('tau_m') * self.sim.ms}

    def set_biases(self):
        if any(self._biases):  # TODO: Implement biases.
            warnings.warn("Biases not implemented.", RuntimeWarning)
