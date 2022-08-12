# -*- coding: utf-8 -*-
"""
Building and simulating spiking neural networks using
`SpiNNaker <http://apt.cs.manchester.ac.uk/projects/SpiNNaker/>`_.

Dependency: `SpyNNaker software
<http://spinnakermanchester.github.io/development/devenv.html>`_

@author: UMan, rbodo, piewchee
"""
import warnings

import numpy as np
import os
import sys
from tensorflow import keras

from snntoolbox.parsing.utils import get_type
from snntoolbox.simulation.target_simulators.pyNN_target_sim import \
    SNN as PYSNN
from snntoolbox.simulation.target_simulators.pyNN_target_sim import \
    get_shape_from_label
from snntoolbox.simulation.utils import build_convolution, \
    build_depthwise_convolution, build_1d_convolution, build_pooling
from snntoolbox.utils.utils import confirm_overwrite


class SNN(PYSNN):

    def scale_weights(self, weights):

        from math import exp
        # This ignores the leak term
        tau_syn_E = self.config.getfloat('cell', 'tau_syn_E')
        tau_syn_I = self.config.getfloat('cell', 'tau_syn_I')
        # just to give a sensible answer if tau_syn_E and I are different
        t = self._dt
        tau = (tau_syn_E + tau_syn_I) / 2
        scale = 10 * t / (tau * (exp(-(t / tau)) + 1))
        print('Weights scaled by a factor of {0}'.format(scale,))
        if isinstance(weights, list):
            weights = [(i, j, weight * scale, delay)
                       for (i, j, weight, delay) in weights]
        elif isinstance(weights, np.ndarray):
            weights = weights * scale
        else:
            raise Exception("Not a valid weight type")
        return weights

    def setup_layers(self, batch_shape):
        '''Iterates over all layers to instantiate them in the simulator.'''

        self.add_input_layer(batch_shape)

        for layer in self.parsed_model.layers[1:]:
            print("Instantiating layer: {}".format(layer.name))
            self.add_layer(layer)

            layer_type = get_type(layer)
            print("Building layer: {}".format(layer.name))
            if layer_type == 'Flatten':
                self.flatten_shapes.append(
                    (layer.name, get_shape_from_label(self.layers[-1].label)))
                self.build_flatten(layer)
                continue
            if layer_type in {'Dense', 'Sparse'}:
                self.build_dense(layer)
            elif layer_type in {'Conv1D', 'Conv2D', 'DepthwiseConv2D',
                                'SparseConv2D', 'SparseDepthwiseConv2D'}:
                self.build_convolution(layer)
                self.data_format = layer.data_format
            elif layer_type in {'MaxPooling2D', 'AveragePooling2D'}:
                self.build_pooling(layer)

            elif layer_type == 'ZeroPadding':
                padding = layer.padding
                if set(padding).issubset((1, (1, 1))):
                    self.change_padding = True
                    return
                else:
                    raise NotImplementedError(
                        "Border_mode {} not supported.".format(padding))

    def add_layer(self, layer):

        # This implementation of ZeroPadding layers assumes symmetric single
        # padding ((1, 1), (1, 1)).
        # Todo: Generalize for asymmetric padding or arbitrary size.
        if 'ZeroPadding' in layer.__class__.__name__:
            return
        if 'Flatten' in layer.__class__.__name__:
            return
        if 'Reshape' in layer.__class__.__name__:
            return
        self.layers.append(self.sim.Population(
            np.prod(layer.output_shape[1:], dtype=np.int).item(),
            self.sim.IF_curr_exp, self.cellparams, label=layer.name))

        self.layers[-1].initialize(v=self.layers[-1].get('v_rest'))

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
        all_weights = layer.get_weights()
        if len(all_weights) == 2:
            weights, biases = all_weights
        elif len(all_weights) == 3:
            weights, biases, masks = all_weights
            weights = weights * masks
            print("Building a Sparse layer having", np.count_nonzero(masks),
                  "non-zero entries in its mask")
        else:
            raise ValueError("Layer {} was expected to contain "
                             "weights, biases and, in rare cases,"
                             "masks.".format(layer.name))
        weights = self.scale_weights(weights)
        print(weights.shape)
        n = int(np.prod(layer.output_shape[1:]) / len(biases))
        biases = np.repeat(biases, n).astype('float64')

        self.set_biases(np.array(biases, 'float64'))
        delay = self.config.getfloat('cell', 'delay')
        if len(self.flatten_shapes) == 1:
            flatten_name, shape = self.flatten_shapes.pop()
            y_in = 1
            if self.data_format == 'channels_last':
                print("Not swapping data_format of Flatten layer.")
                if len(shape) == 2:
                    x_in, f_in = shape
                    #weights = weights.flatten()
                else:
                    y_in, x_in, f_in = shape
                '''output_neurons = weights.shape[1]
                weights = weights.reshape((x_in, y_in, f_in, output_neurons), order ='C')
                weights = np.rollaxis(weights, 1, 0)
                weights = weights.reshape((y_in*x_in*f_in, output_neurons), order ='C')
                '''
            else:
                print("Swapping data_format of Flatten layer.")
                if len(shape) == 3:
                    f_in, y_in, x_in = shape
                    output_neurons = weights.shape[1]
                    weights = weights.reshape(
                        (y_in, x_in, f_in, output_neurons), order='F')
                    weights = np.rollaxis(weights, 2, 0)
                    weights = weights.reshape(
                        (y_in * x_in * f_in, output_neurons), order='F')
                elif len(shape) == 2:
                    f_in, x_in = shape
                    weights = np.rollaxis(weights, 1, 0)
                    #weights = np.flatten(weights)
                else:
                    print(
                        "The input weight matrix did not have the expected dimesnions")
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
                    if c[2] > 0.0:
                        exc_connections.append(c)
                    elif c[2] < 0.0:
                        inh_connections.append(c)
        elif len(self.flatten_shapes) > 1:
            raise RuntimeWarning("Not all Flatten layers have been consumed.")
        else:
            exc_connections = [(i, j, weights[i, j], delay)
                               for i, j in zip(*np.nonzero(weights > 0))]
            inh_connections = [(i, j, weights[i, j], delay)
                               for i, j in zip(*np.nonzero(weights < 0))]

        if self.config.getboolean('tools', 'simulate'):
            self.connections.append(self.sim.Projection(
                self.layers[-2], self.layers[-1],
                self.sim.FromListConnector(exc_connections,
                                           ['weight', 'delay']),
                receptor_type='excitatory',
                label=self.layers[-1].label + '_excitatory'))

            self.connections.append(self.sim.Projection(
                self.layers[-2], self.layers[-1],
                self.sim.FromListConnector(inh_connections,
                                           ['weight', 'delay']),
                receptor_type='inhibitory',
                label=self.layers[-1].label + '_inhibitory'))
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

    def build_convolution(self, layer):

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

        if get_type(layer) in ['Conv2D', 'SparseConv2D']:
            weights, biases = build_convolution(layer, delay, transpose_kernel)
        elif get_type(layer) in ['DepthwiseConv2D', 'SparseDepthwiseConv2D']:
            weights, biases = build_depthwise_convolution(
                layer, delay, transpose_kernel)
        elif get_type(layer) == 'Conv1D':
            weights, biases = build_1d_convolution(layer, delay)
        else:
            ValueError("Layer {} of type {} unrecognised here. "
                       "How did you get into this function?".format(
                           layer.name, get_type(layer)
                       ))

        self.set_biases(biases)
        weights = self.scale_weights(weights)

        exc_connections = [c for c in weights if c[2] > 0]
        inh_connections = [c for c in weights if c[2] < 0]

        if self.config.getboolean('tools', 'simulate'):
            self.connections.append(self.sim.Projection(
                self.layers[-2], self.layers[-1],
                (self.sim.FromListConnector(exc_connections, ['weight', 'delay'])),
                receptor_type='excitatory',
                label=self.layers[-1].label + '_excitatory'))

            self.connections.append(self.sim.Projection(
                self.layers[-2], self.layers[-1],
                (self.sim.FromListConnector(inh_connections, ['weight', 'delay'])),
                receptor_type='inhibitory',
                label=self.layers[-1].label + '_inhibitory'))
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

    def build_pooling(self, layer):

        delay = self.config.getfloat('cell', 'delay')

        weights = build_pooling(layer, delay)
        weights = self.scale_weights(weights)
        if self.config.getboolean('tools', 'simulate'):
            self.connections.append(self.sim.Projection(
                self.layers[-2], self.layers[-1],
                self.sim.FromListConnector(weights,
                                           ['weight', 'delay']),
                receptor_type='excitatory',
                label=self.layers[-1].label + '_excitatory'))
        else:
            # The spinnaker implementation of Projection.save() is not working
            # yet, so we do save the connections manually here.
            filepath = os.path.join(self.config.get('paths', 'path_wd'),
                                    self.layers[-1].label)
            # noinspection PyTypeChecker
            np.savetxt(filepath, np.array(connections),
                       ['%d', '%d', '%.18f', '%.3f'],
                       header="columns = ['i', 'j', 'weight', 'delay']")

    def save(self, path, filename):

        # Temporary fix to stop IsADirectory error
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
        self.sim.set_number_of_neurons_per_core(
            self.sim.IF_curr_exp, self.config.getfloat(
                'spinnaker', 'number_of_neurons_per_core'))
        data = kwargs[str('x_b_l')]
        if self.data_format == 'channels_last' and data.ndim == 4:
            data = np.moveaxis(data, 3, 1)

        x_flat = np.ravel(data)
        if self._poisson_input:
            rates = 1000 * x_flat / self.rescale_fac
            self.layers[0].set(rate=rates)
        elif self._is_aedat_input:
            raise NotImplementedError
        else:
            spike_times = \
                [np.linspace(0, self._duration, self._duration * amplitude)
                 for amplitude in x_flat]
            self.layers[0].set(spike_times=spike_times)
        import pylab
        current_time = pylab.datetime.datetime.now().strftime("_%H%M%S_%d%m%Y")

        runlabel = self.config.get("paths", "runlabel")

        try:
            from pynn_object_serialisation.functions import intercept_simulator
            intercept_simulator(
                self.sim,
                runlabel + "_serialised",
                post_abort=False,
                custom_params={
                    'runtime': self._duration})
        except Exception:
            print("There was a problem with serialisation.")
        if self.config.getboolean('tools', 'serialise_only'):
            sys.exit('finished after serialisation')
        self.sim.run(self._duration)
        print("\nCollecting results...")
        output_b_l_t = self.get_recorded_vars(self.layers)

        return output_b_l_t

    def get_spiketrains_input(self):
        shape = list(self.parsed_model.input_shape) + [self._num_timesteps]
        spiketrains_flat = self.layers[0].get_data(
            'spikes').segments[-1].spiketrains
        spiketrains_b_l_t = self.reshape_flattened_spiketrains(
            spiketrains_flat, shape)
        return spiketrains_b_l_t

    def get_spiketrains_output(self):
        shape = [self.batch_size, self.num_classes, self._num_timesteps]
        spiketrains_flat = self.layers[-1].get_data(
            'spikes').segments[-1].spiketrains
        spiketrains_b_l_t = self.reshape_flattened_spiketrains(
            spiketrains_flat, shape)
        return spiketrains_b_l_t

    def get_spiketrains(self, **kwargs):
        # There is an overhead associated with retrieving data on SpiNNaker
        # and so here only the spikes are got
        j = self._spiketrains_container_counter
        if self.spiketrains_n_b_l_t is None \
                or j >= len(self.spiketrains_n_b_l_t):
            return None

        shape = self.spiketrains_n_b_l_t[j][0].shape

        # Outer for-loop that calls this function starts with
        # 'monitor_index' = 0, but this is reserved for the input and handled
        # by `get_spiketrains_input()`.
        i = kwargs[str('monitor_index')]
        if i == 0:
            return
        spiketrains_flat = self.layers[i].get_data(
            'spikes').segments[-1].spiketrains
        spiketrains_b_l_t = self.reshape_flattened_spiketrains(
            spiketrains_flat, shape)
        return spiketrains_b_l_t

    def get_vmem(self, **kwargs):
        # There is an overhead associated with retrieving data on SpiNNaker
        # and so here only the membrane voltages are got
        i = kwargs[str('monitor_index')]
        try:
            vs = self.layers[i].get_data('v').segments[-1].analogsignals
        except Exception:
            return None

        if len(vs) > 0:
            return np.array([np.swapaxes(v, 0, 1) for v in vs])
