# -*- coding: utf-8 -*-
"""
Building and simulating spiking neural networks using
`pyNN <http://neuralensemble.org/docs/PyNN/>`_.

@author: rbodo
"""

from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import os
import warnings

import numpy as np
from future import standard_library
from six.moves import cPickle

from snntoolbox.utils.utils import confirm_overwrite
from snntoolbox.simulation.utils import AbstractSNN

standard_library.install_aliases()

cellparams_pyNN = {'v_thresh', 'v_reset', 'v_rest', 'e_rev_E', 'e_rev_I', 'cm',
                   'i_offset', 'tau_refrac', 'tau_m', 'tau_syn_E', 'tau_syn_I'}


def connect(f):
    """Connect layers."""

    from functools import wraps

    @wraps(f)
    def wrapper(self):
        f(self)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            warnings.warn('deprecated', UserWarning)

            self.connections.append(self.sim.Projection(
                self.layers[-2], self.layers[-1],
                self.sim.FromListConnector(self._conns, ['weight', 'delay'])))
        return wrapper


class SNN(AbstractSNN):
    """Class to hold the compiled spiking neural network.

    Represents the compiled spiking neural network, ready for testing in a
    spiking simulator.

    Attributes
    ----------

    layers: list[pyNN.Population]
        Each entry represents a layer, i.e. a population of neurons, in form of
        pyNN ``Population`` objects.

    connections: list[pyNN.Projection]
        pyNN ``Projection`` objects representing the connections between
        individual layers.

    cellparams: dict
        Neuron cell parameters determining properties of the spiking neurons in
        pyNN simulators.
    """

    def __init__(self, config, queue=None):

        AbstractSNN.__init__(self, config, queue)

        self.layers = []
        self._conns = []  # Temporary container for layer connections.
        self._biases = []  # Temporary container for layer biases.
        self.connections = []  # Final container for all layers.
        self.cellparams = {key: config.getfloat('cell', key) for key in
                           cellparams_pyNN}

    @property
    def is_parallelizable(self):
        return False

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
        from snntoolbox.simulation.utils import build_convolution

        delay = self.config.getfloat('cell', 'delay')
        self._conns, self._biases = build_convolution(layer, delay)
        self.set_biases()

    @connect
    def build_pooling(self, layer):
        from snntoolbox.simulation.utils import build_pooling

        delay = self.config.getfloat('cell', 'delay')
        self._conns = build_pooling(layer, delay)

    def compile(self):

        pass

    def simulate(self, **kwargs):

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

        output_b_l_t = self.get_recorded_vars(self.layers)

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
        """Set biases.

        Notes
        -----

        This has not been tested yet.
        """

        self.layers[-1].set(i_offset=self._biases)

    def get_vars_to_record(self):
        """Get variables to record during simulation.

        Returns
        -------

        vars_to_record: list[str]
            The names of variables to record during simulation.
        """

        vars_to_record = []

        if any({'spiketrains', 'spikerates', 'correlation', 'spikecounts',
                'hist_spikerates_activations'} & self._plot_keys) \
                or 'spiketrains_n_b_l_t' in self._log_keys:
            vars_to_record.append('spikes')

        if 'mem_n_b_l_t' in self._log_keys or 'mem' in self._plot_keys:
            vars_to_record.append('v')

        return vars_to_record

    def get_spiketrains(self, **kwargs):
        j = self._spiketrains_container_counter
        if j >= len(self.spiketrains_n_b_l_t):
            return None

        shape = self.spiketrains_n_b_l_t[j][0].shape

        # Outer for-loop that calls this function starts with
        # 'monitor_index' = 0, but this is reserved for the input and handled by
        # `get_spiketrains_input()`.
        i = len(self.layers) - 1 if kwargs['monitor_index'] == -1 else \
            kwargs['monitor_index'] + 1
        spiketrains_flat = self.layers[i].get_data().segments[-1].spiketrains
        spiketrains_b_l_t = self.reshape_flattened_spiketrains(spiketrains_flat,
                                                               shape)
        return spiketrains_b_l_t

    def get_spiketrains_input(self):
        shape = list(self.parsed_model.input_shape) + [self._num_timesteps]
        spiketrains_flat = self.layers[0].get_data().segments[-1].spiketrains
        spiketrains_b_l_t = self.reshape_flattened_spiketrains(spiketrains_flat,
                                                               shape)
        return spiketrains_b_l_t

    def get_vmem(self, **kwargs):
        vs = kwargs['layer'].get_data().segments[-1].analogsignalarrays
        return np.array([np.array(v) for v in vs])

    def save_assembly(self, path, filename):
        """Write layers of neural network to disk.

        The size, structure, labels of all the population of an assembly are
        stored in a dictionary such that one can load them again using the
        `load_assembly` function.

        The term "assembly" refers to pyNN internal nomenclature, where
        ``Assembly`` is a collection of layers (``Populations``), which in turn
        consist of a number of neurons (``cells``).

        Parameters
        ----------

        path: str
            Path to directory where to save layers.

        filename: str
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

        path: str
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
        `save_assembly` function.

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

    def set_spiketrain_stats_input(self):
        AbstractSNN.set_spiketrain_stats_input(self)
