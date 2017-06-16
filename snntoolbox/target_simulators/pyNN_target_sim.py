# -*- coding: utf-8 -*-
"""Building SNNs using pyNN simulator.

The modules in ``target_simulators`` package allow building a spiking network
and exporting it for use in a spiking simulator.

This particular module offers functionality for pyNN simulators Brian, Nest,
Neuron. Adding another simulator requires implementing the class
``AbstractSNN`` with its methods tailored to the specific simulator.

Created on Thu May 19 15:00:02 2016

@author: rbodo
"""

from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import os
import warnings
import numpy as np

from future import standard_library
from six.moves import cPickle
from snntoolbox.io_utils.common import confirm_overwrite
from snntoolbox.target_simulators.common import AbstractSNN

standard_library.install_aliases()

cellparams_pyNN = {'v_thresh', 'v_reset', 'v_rest', 'e_rev_E', 'e_rev_I', 'cm',
                   'i_offset', 'tau_refrac', 'tau_m', 'tau_syn_E', 'tau_syn_I'}


class SNN(AbstractSNN):
    """Class to hold the compiled spiking neural network.

    Class to hold the compiled spiking neural network, ready for testing in a
    spiking simulator.

    Attributes
    ----------

    layers: list
        Each entry represents a layer, i.e. a population of neurons, in form of
        pyNN ``Population`` objects.

    connections: list
        pyNN ``Projection`` objects representing the connections between
        individual layers.

    cellparams: dict
        Neuron cell parameters determining properties of the spiking neurons in
        pyNN simulators.

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
        self.cellparams = {key: config.getfloat('cell', key) for key in
                           cellparams_pyNN}

    def add_input_layer(self, input_shape):

        self.layers.append(self.sim.Population(
            np.asscalar(np.prod(input_shape[1:], dtype=np.int)),
            self.sim.SpikeSourcePoisson(), label='InputLayer'))

    def add_layer(self, layer):

        if 'Flatten' in layer.__class__.__name__:
            return

        self._conns = []
        self.layers.append(self.sim.Population(
            np.asscalar(np.prod(layer.output_shape[1:], dtype=np.int)),
            self.sim.IF_cond_exp, self.cellparams, label=layer.name))

    @connect
    def build_dense(self, layer):

        if layer.activation == 'softmax':
            raise warnings.warn("Activation 'softmax' not implemented. Using "
                                "'relu' activation instead.", RuntimeWarning)

        weights, self._biases = layer.get_weights()
        self.set_biases()
        delay = self.config.getfloat('cell', 'delay')
        for i in range(len(weights)):
            for j in range(len(weights[0])):
                self._conns.append((i, j, weights[i, j], delay))

    @connect
    def build_convolution(self, layer):
        from snntoolbox.target_simulators.common import build_convolution

        delay = self.config.getfloat('cell', 'delay')
        self._conns, self._biases = build_convolution(layer, delay)
        self.set_biases()

    @connect
    def build_pooling(self, layer):
        from snntoolbox.target_simulators.common import build_pooling

        delay = self.config.getfloat('cell', 'delay')
        self._conns = build_pooling(layer, delay)

    def compile(self):

        pass

    def simulate(self, **kwargs):

        from snntoolbox.io_utils.plotting import plot_potential
        from snntoolbox.core.util import get_layer_ops

        if self._poisson_input:
            rates = kwargs['x_b_l'].flatten()
            for neuron_idx, neuron in enumerate(self.layers[0]):
                neuron.rate = rates[neuron_idx] / self.rescale_fac
        elif self._dataset_format == 'aedat':
            raise NotImplementedError
        else:
            constant_input_currents = kwargs['x_b_l'].flatten()
            try:
                for neuron_idx, neuron in enumerate(self.layers[0]):
                    # TODO: Implement constant input currents.
                    neuron.current = constant_input_currents[neuron_idx]
            except AttributeError:
                raise NotImplementedError

        self.sim.run(self._duration)

        # Get spiketrains of output layer.
        out_spikes = self.layers[-1].get_data().segments[-1].spiketrains

        # For each time step, get number of spikes of all neurons in the output
        # layer.
        output_b_l_t = np.zeros((self.batch_size, self.num_classes,
                                 self._num_timesteps), 'int32')
        for k, spiketrain in enumerate(out_spikes):
            for t in range(len(self._num_timesteps)):
                output_b_l_t[0, k, t] = np.count_nonzero(spiketrain <=
                                                         t * self._dt)

        # Record neuron variables.
        i = 0
        for layer in self.layers[1:]:

            # Get spike trains.
            try:
                spiketrains = layer.get_data().segments[-1].spiketrains
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
            i += 1

        i = 0
        for layer in self.layers[1:]:

            # Get membrane potentials.
            try:
                mem = np.array([
                    np.array(v) for v in
                    layer.get_data().segments[-1].analogsignalarrays])
            except AttributeError:
                continue

            # Reshape flat array to original layer shape.
            layer_shape = self.mem_n_b_l_t[i][0].shape
            self.mem_n_b_l_t[i][0] = np.reshape(mem, layer_shape)

            # Plot membrane potentials of layer.
            times = self._dt * np.arange(self._num_timesteps)
            show_legend = True if i >= len(self.layers) - 2 else False
            plot_potential(times, self.mem_n_b_l_t[i], self.config, show_legend,
                           self.config['paths']['log_dir_of_current_run'])
            i += 1

        # Get spike trains of input layer.
        spiketrains = self.layers[0].get_data().segments[-1].spiketrains

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
            self.sim.reset()
            print("Done.")

    def end_sim(self):

        self.sim.end()

    def save(self, path, filename):

        print("Saving model to {}...".format(path))
        self.save_assembly(path, filename)
        self.save_connections(path)
        print("Done.\n")

    def load(self, path, filename):

        self.layers = self.load_assembly(path, filename)
        for i in range(len(self.layers)-1):
            filepath = os.path.join(path, self.layers[i+1].label)
            assert os.path.isfile(filepath), \
                "Connections were not found at specified location."
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                warnings.warn('deprecated', UserWarning)
                self.sim.Projection(self.layers[i], self.layers[i+1],
                                    self.sim.FromFileConnector(filepath))

    def init_cells(self):

        vars_to_record = self.get_vars_to_record()

        if 'spikes' in vars_to_record:
            self.layers[0].record(['spikes'])  # Input layer has no 'v'

        for layer in self.layers[1:]:
            layer.set(**self.cellparams)
            layer.initialize(v=self.layers[1].get('v_rest'))
            layer.record(vars_to_record)

        # The spikes of the last layer are recorded by default because they
        # contain the networks output (classification guess).
        if 'spikes' not in vars_to_record:
            vars_to_record.append('spikes')
        self.layers[-1].record(vars_to_record)

    def set_biases(self):
        self.layers[-1].set(i_offset=self._biases)

    def get_vars_to_record(self):

        vars_to_record = []

        if any({'spiketrains', 'spikerates', 'correlation', 'spikecounts',
                'hist_spikerates_activations'} & self._plot_keys) \
                or 'spiketrains_n_b_l_t' in self._log_keys:
            vars_to_record.append('spikes')

        if 'mem_n_b_l_t' in self._log_keys or 'mem' in self._plot_keys:
            vars_to_record.append('v')

        return vars_to_record

    def save_assembly(self, path, filename):
        """Write layers of neural network to disk.

        The size, structure, labels of all the population of an assembly are
        stored in a dictionary such that one can load them again using the
        ``load_assembly`` function.

        The term "assembly" refers to pyNN internal nomenclature, where
        ``Assembly`` is a collection of layers (``Populations``), which in turn
        consist of a number of neurons (``cells``).

        Parameters
        ----------

        path: string
            Path to directory where to save layers.

        filename: string, optional
            Name of file to write layers to.
        """

        filepath = os.path.join(path, filename)

        if not (self.config.getboolean('output', 'overwrite') or
                confirm_overwrite(filepath)):
            return

        print("Saving assembly...")

        s = {}
        labels = []
        variables = ['size', 'structure', 'label']
        for population in self.layers:
            labels.append(population.label)
            data = {}
            for variable in variables:
                data[variable] = getattr(population, variable)
            data['celltype'] = population.celltype.describe()
            if population.label != 'InputLayer':
                data['i_offset'] = population.get('i_offset')
            s[population.label] = data
        s['labels'] = labels  # List of population labels describing the net.
        s['variables'] = variables  # List of variable names.
        s['size'] = len(self.layers)  # Number of populations in assembly.
        cPickle.dump(s, open(filepath, 'wb'))

    def save_connections(self, path):
        """Write parameters of a neural network to disk.

        The parameters between two layers are saved in a text file.
        They can then be used to connect pyNN populations e.g. with
        ``sim.Projection(layer1, layer2, sim.FromListConnector(filename))``,
        where ``sim`` is a simulator supported by pyNN, e.g. Brian, NEURON, or
        NEST.

        Parameters
        ----------

        path: string
            Path to directory where connections are saved.

        Return
        ------
            Text files containing the layer connections. Each file is named
            after the layer it connects to, e.g. ``layer2.txt`` if connecting
            layer1 to layer2.
        """

        print("Saving connections...")

        # Iterate over layers to save each projection in a separate txt file.
        for projection in self.connections:
            filepath = os.path.join(path, projection.label.partition('→')[-1])
            if self.config.getboolean('output', 'overwrite') or \
                    confirm_overwrite(filepath):
                projection.save('connections', filepath)

    def load_assembly(self, path, filename):
        """Load the populations in an assembly.

        Loads the populations in an assembly that was saved with the
        ``save_assembly`` function.

        The term "assembly" refers to pyNN internal nomenclature, where
        ``Assembly`` is a collection of layers (``Populations``), which in turn
        consist of a number of neurons (``cells``).

        Parameters
        ----------

        path: str
            Path to directory where to load model from.

        filename: str
            Name of file to load model from.

        Returns
        -------

        layers: list[pyNN.Population]
            List of pyNN ``Population`` objects.
        """

        import sys

        filepath = os.path.join(path, filename)
        assert os.path.isfile(filepath), \
            "Spiking neuron layers were not found at specified location."
        if sys.version_info < (3,):
            s = cPickle.load(open(filepath, 'rb'))
        else:
            s = cPickle.load(open(filepath, 'rb'), encoding='bytes')

        # Iterate over populations in assembly
        layers = []
        for label in s['labels']:
            celltype = getattr(self.sim, s[label]['celltype'])
            population = self.sim.Population(s[label]['size'], celltype,
                                             celltype.default_parameters,
                                             structure=s[label]['structure'],
                                             label=label)
            # Set the rest of the specified variables, if any.
            for variable in s['variables']:
                if getattr(population, variable, None) is None:
                    setattr(population, variable, s[label][variable])
            if label != 'InputLayer':
                population.set(i_offset=s[label]['i_offset'])
            layers.append(population)

        return layers


def connect(f):
    """Connect layers."""

    def wrapper(self):
        f(self)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            warnings.warn('deprecated', UserWarning)

            self.connections.append(self.sim.Projection(
                self.layers[-2], self.layers[-1],
                self.sim.FromListConnector(self._conns,
                                           ['weight', 'delay'])))
        return wrapper
