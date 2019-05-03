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

        self.set_biases(np.array(biases, 'float64'))
        delay = self.config.getfloat('cell', 'delay')

        if len(self.flatten_shapes) == 1:
            print("Swapping data_format of Flatten layer.")
            flatten_name, shape = self.flatten_shapes.pop()
            if self.data_format == 'channels_last':
                y_in, x_in, f_in = shape
            else:
                f_in, y_in, x_in = shape
            exc_connections = []
            inh_connections = []
            for i in range(weights.shape[0]):  # Input neurons
                # Sweep across channel axis of feature map. Assumes that each
                # consecutive input neuron lies in a different channel. This is
                # the case for channels_last, but not for channels_first.
                f = i % f_in
                # Sweep across height of feature map. Increase y by one if all
                # rows along the channel axis were seen.
                y = i // (f_in * x_in)
                # Sweep across width of feature map.
                x = (i // f_in) % x_in
                new_i = f * x_in * y_in + x_in * y + x
                for j in range(weights.shape[1]):  # Output neurons
                    c = (new_i, j, weights[i, j], delay)
                    if c[2] > 0:
                        exc_connections.append(c)
                    else:
                        inh_connections.append(c)
        elif len(self.flatten_shapes) > 1:
            raise RuntimeWarning("Not all Flatten layers have been consumed.")
        else:
            exc_connections = [(i, j, weights[i, j], delay)
                               for i, j in zip(*np.nonzero(weights > 0))]
            inh_connections = [(i, j, weights[i, j], delay)
                               for i, j in zip(*np.nonzero(weights <= 0))]

        if self.config.getboolean('tools', 'simulate'):
            self.connections.append(self.sim.Projection(
                self.layers[-2], self.layers[-1],
                self.sim.FromListConnector(exc_connections,
                                           ['weight', 'delay']),
                label=self.layers[-1].label+'_excitatory'))

            self.connections.append(self.sim.Projection(
                self.layers[-2], self.layers[-1],
                self.sim.FromListConnector(inh_connections,
                                           ['weight', 'delay']),
                receptor_type='inhibitory',
                label=self.layers[-1].label+'_inhibitory'))
        else:
            # The spinnaker implementation of Projection.save() is not working
            # yet, so we do save the connections manually here.
            filepath = os.path.join(self.config.get('paths', 'path_wd'),
                                    self.layers[-1].label)
            # noinspection PyTypeChecker
            np.savetxt(filepath + '_excitatory', np.array(exc_connections),
                       ['%d', '%d', '%.18f', '%.3f'],
                       header="columns = ['i', 'j', 'weight', 'delay']")
            # noinspection PyTypeChecker
            np.savetxt(filepath + '_inhibitory', np.array(inh_connections),
                       ['%d', '%d', '%.18f', '%.3f'],
                       header="columns = ['i', 'j', 'weight', 'delay']")

        lines = [
            "\n",
            "\t# Load dense projections created by snntoolbox.\n",
            "\tfilepath = os.path.join(path_wd, layers[-1].label + "
            "'_excitatory')"
            "\n",
            "\tsim.Projection(layers[-2], layers[-1], sim.FromFileConnector("
            "filepath))\n",
            "\tfilepath = os.path.join(path_wd, layers[-1].label + "
            "'_inhibitory')"
            "\n",
            "\tsim.Projection(layers[-2], layers[-1], sim.FromFileConnector("
            "filepath), receptor_type='inhibitory')\n",
            "\n",
            "\t# Set biases.\n",
            "\tfilepath = os.path.join(path_wd, layers[-1].label + '_biases')"
            "\n",
            "\tbiases = np.loadtxt(filepath)\n",
            "\tlayers[-1].set(i_offset=biases*dt/1e2)\n"
        ]
        with open(self.output_script_path, 'a') as f:
            f.writelines(lines)

    def build_convolution(self, layer):
        from snntoolbox.simulation.utils import build_convolution

        # If the parsed model contains a ZeroPadding layer, we need to tell the
        # Conv layer about it here, because ZeroPadding layers are removed when
        # building the pyNN model.
        if self.change_padding:
            if layer.padding == 'valid':
                self.change_padding = False
                layer.padding = 'ZeroPadding'
            else:
                raise NotImplementedError(
                    "Border_mode {} in combination with ZeroPadding is not "
                    "supported.".format(layer.padding))

        delay = self.config.getfloat('cell', 'delay')
        transpose_kernel = \
            self.config.get('simulation', 'keras_backend') == 'tensorflow'
        weights, biases = build_convolution(layer, delay, transpose_kernel)
        self.set_biases(biases)

        exc_connections = [c for c in weights if c[2] > 0]
        inh_connections = [c for c in weights if c[2] <= 0]

        if self.config.getboolean('tools', 'simulate'):
            self.connections.append(self.sim.Projection(
                self.layers[-2], self.layers[-1],
                self.sim.FromListConnector(exc_connections,
                                           ['weight', 'delay']),
                receptor_type='excitatory',
                label=self.layers[-1].label+'_excitatory'))

            self.connections.append(self.sim.Projection(
                self.layers[-2], self.layers[-1],
                self.sim.FromListConnector(inh_connections,
                                           ['weight', 'delay']),
                receptor_type='inhibitory',
                label=self.layers[-1].label+'_inhibitory'))
        else:
            # The spinnaker implementation of Projection.save() is not working
            # yet, so we do save the connections manually here.
            filepath = os.path.join(self.config.get('paths', 'path_wd'),
                                    self.layers[-1].label)
            # noinspection PyTypeChecker
            np.savetxt(filepath + '_excitatory', np.array(exc_connections),
                       ['%d', '%d', '%.18f', '%.3f'],
                       header="columns = ['i', 'j', 'weight', 'delay']")
            # noinspection PyTypeChecker
            np.savetxt(filepath + '_inhibitory', np.array(inh_connections),
                       ['%d', '%d', '%.18f', '%.3f'],
                       header="columns = ['i', 'j', 'weight', 'delay']")

        lines = [
            "\n",
            "\t# Load convolution projections created by snntoolbox.\n",
            "\tfilepath = os.path.join(path_wd, layers[-1].label + "
            "'_excitatory')"
            "\n",
            "\tsim.Projection(layers[-2], layers[-1], sim.FromFileConnector("
            "filepath))\n",
            "\tfilepath = os.path.join(path_wd, layers[-1].label + "
            "'_inhibitory')"
            "\n",
            "\tsim.Projection(layers[-2], layers[-1], sim.FromFileConnector("
            "filepath), receptor_type='inhibitory')\n",
            "\n",
            "\t# Set biases.\n",
            "\tfilepath = os.path.join(path_wd, layers[-1].label + '_biases')"
            "\n",
            "\tbiases = np.loadtxt(filepath)\n",
            "\tlayers[-1].set(i_offset=biases*dt/1e2)\n"
        ]
        with open(self.output_script_path, 'a') as f:
            f.writelines(lines)

    def build_pooling(self, layer):
        from snntoolbox.simulation.utils import build_pooling

        delay = self.config.getfloat('cell', 'delay')
        transpose_kernel = \
            self.config.get('simulation', 'keras_backend') == 'tensorflow'
        weights = build_pooling(layer, delay) 
        if self.config.getboolean('tools', 'simulate'):
            self.connections.append(self.sim.Projection(
                self.layers[-2], self.layers[-1],
                self.sim.FromListConnector(weights,
                                           ['weight', 'delay']),
                receptor_type='excitatory',
                label=self.layers[-1].label+'_excitatory'))
        else:
            # The spinnaker implementation of Projection.save() is not working
            # yet, so we do save the connections manually here.
            filepath = os.path.join(self.config.get('paths', 'path_wd'),
                                    self.layers[-1].label)
            # noinspection PyTypeChecker
            np.savetxt(filepath, np.array(connections),
                       ['%d', '%d', '%.18f', '%.3f'],
                       header="columns = ['i', 'j', 'weight', 'delay']")

        lines = [
            "\n",
            "\t# Load pooling projections created by snntoolbox.\n",
            "\tfilepath = os.path.join(path_wd, layers[-1].label)\n",
            "\tsim.Projection(layers[-2], layers[-1], sim.FromFileConnector("
            "filepath))\n"
        ]
        with open(self.output_script_path, 'a') as f:
            f.writelines(lines)
    
    def save(self, path, filename):
        #Temporary fix to stop IsADirectory error 
        print("Not saving model to {}...".format(path))
    
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
            filepath = os.path.join(path, projection._projection_edge.label)
            if self.config.getboolean('output', 'overwrite') or \
                    confirm_overwrite(filepath):
                projection.save('connections', filepath)
                
    def simulate(self, **kwargs):
        #sim.set_number_of_neurons_per_core
        data = kwargs[str('x_b_l')]
        if self.data_format == 'channels_last' and data.ndim == 4:
            data = np.moveaxis(data, 3, 1)

        x_flat = np.ravel(data)
        if self._poisson_input:
            self.layers[0].set(rate=list(x_flat / self.rescale_fac * 1000))
        elif self._dataset_format == 'aedat':
            raise NotImplementedError
        else:
            spike_times = \
                [np.linspace(0, self._duration, self._duration * amplitude)
                 for amplitude in x_flat]
            self.layers[0].set(spike_times=spike_times)
        self.sim.run(self._duration - self._dt)
        
        print("\nCollecting results...")
        output_b_l_t = self.get_recorded_vars(self.layers)

        return output_b_l_t
