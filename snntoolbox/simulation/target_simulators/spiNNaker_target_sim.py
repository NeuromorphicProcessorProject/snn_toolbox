# -*- coding: utf-8 -*-
"""
Building and simulating spiking neural networks using
`SpiNNaker <http://apt.cs.manchester.ac.uk/projects/SpiNNaker/>`_.

Dependency: `SpyNNaker software
<http://spinnakermanchester.github.io/development/devenv.html>`_
Some changes have to be made in SpyNNaker script due to compatibilty.
(@piewchee: Please specify.)

@author: rbodo, piewchee
"""

import os
import warnings

import numpy as np

from snntoolbox.utils.utils import confirm_overwrite
from snntoolbox.simulation.target_simulators.pyNN_target_sim import SNN as PYSNN


class SNN(PYSNN):

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

        weights, biases = layer.get_weights()
        biases = np.array(biases, 'float64')
        self.set_biases(biases)
        delay = self.config.getfloat('cell', 'delay')

        exc_connections = [(i, j, weights[i, j], delay)
                           for i, j in zip(*np.nonzero(weights > 0))]
        inh_connections = [(i, j, weights[i, j], delay)
                           for i, j in zip(*np.nonzero(weights <= 0))]

        self.connections.append(self.sim.Projection(
            self.layers[-2], self.layers[-1],
            self.sim.FromListConnector(exc_connections, ['weight', 'delay'])))

        self.connections.append(self.sim.Projection(
            self.layers[-2], self.layers[-1],
            self.sim.FromListConnector(inh_connections, ['weight', 'delay']),
            receptor_type='inhibitory'))

    def build_convolution(self, layer):
        from snntoolbox.simulation.utils import build_convolution

        delay = self.config.getfloat('cell', 'delay')
        transpose_kernel = \
            self.config.get('simulation', 'keras_backend') == 'tensorflow'
        connections, biases = build_convolution(layer, delay, transpose_kernel)

        self.set_biases(biases)

        exc_connections = [c for c in connections if c[2] > 0]
        inh_connections = [c for c in connections if c[2] <= 0]

        self.connections.append(self.sim.Projection(
            self.layers[-2], self.layers[-1],
            self.sim.FromListConnector(exc_connections, ['weight', 'delay'])))

        self.connections.append(self.sim.Projection(
            self.layers[-2], self.layers[-1],
            self.sim.FromListConnector(inh_connections, ['weight', 'delay']),
            receptor_type='inhibitory'))

    def build_pooling(self, layer):
        from snntoolbox.simulation.utils import build_pooling

        delay = self.config.getfloat('cell', 'delay')
        connections = build_pooling(layer, delay)
        self.connections.append(self.sim.Projection(
            self.layers[-2], self.layers[-1],
            self.sim.FromListConnector(connections, ['weight', 'delay'])))

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
            # filename = projection.label.partition('→')[-1] \
            #     if hasattr(projection, 'label') else 'layer_{}'.format(i)
            # filepath =  os.path.join(path, filename)
            # filepath = os.path.join(path, self.layers[i + 1].label)
            filepath = os.path.join(path, projection.label.partition('→')[-1])
            if self.config.getboolean('output', 'overwrite') or \
                    confirm_overwrite(filepath):
                projection.save('connections', filepath)
