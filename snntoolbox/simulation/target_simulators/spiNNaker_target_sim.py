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
import keras
from snntoolbox.utils.utils import confirm_overwrite
from snntoolbox.simulation.target_simulators.pyNN_target_sim import SNN as PYSNN


class SNN(PYSNN):

    def setup_layers(self, batch_shape):
        '''Iterates over all layers to instantiate them in the simulator.
        This is done in reverse for SpiNNaker because it changes
         machine placement so larger layers are placed later.'''
        from snntoolbox.parsing.utils import get_type
        from snntoolbox.simulation.target_simulators.pyNN_target_sim import get_shape_from_label

        self.add_input_layer(batch_shape)

        # temp_layers = self.layers
        # self.layers = []
        # "Adding the input layer"
        # self.layers.append(temp_layers.pop())

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
            if layer_type == 'Dense':
                self.build_dense(layer)
            elif layer_type in {'Conv2D','DepthwiseConv2D'}:
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
            np.asscalar(np.prod(layer.output_shape[1:], dtype=np.int)),
            self.sim.IF_curr_exp, self.cellparams, label=layer.name))

        self.layers[-1].initialize(v=self.layers[-1].get('v_rest'))
        

        lines = [
            "\n",
            "\t# Add layer {}.\n".format(layer.name),
            "\tprint('Building layer {}.')\n".format(layer.name),
            "\tlayers.append(sim.Population(np.asscalar(np.prod({}, "
            "dtype=np.int)), sim.IF_curr_exp, cellparams, label='{}'))\n"
            "".format(layer.output_shape[1:], layer.name),
            "\tlayers[-1].initialize(v=layers[-1].get('v_rest'))\n"
        ]
        with open(self.output_script_path, 'a') as f:
            f.writelines(lines)

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
            flatten_name, shape = self.flatten_shapes.pop() 
            if self.data_format == 'channels_last':
                print("Not swapping data_format of Flatten layer.")
                y_in, x_in, f_in = shape
                '''output_neurons = weights.shape[1]
                weights = weights.reshape((x_in, y_in, f_in, output_neurons), order ='C')
                weights = np.rollaxis(weights, 1, 0)
                weights = weights.reshape((y_in*x_in*f_in, output_neurons), order ='C')
                '''
            else:
                print("Swapping data_format of Flatten layer.")
                f_in, y_in, x_in = shape
                output_neurons = weights.shape[1]
                weights = weights.reshape((y_in, x_in, f_in, output_neurons), order='F')
                weights = np.rollaxis(weights, 2, 0)
                weights = weights.reshape((y_in*x_in*f_in, output_neurons), order='F')

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
        from snntoolbox.simulation.utils import build_convolution, build_depthwise_convolution
        from snntoolbox.parsing.utils import get_type

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
        
        if get_type(layer) == 'Conv2D':
            weights, biases = build_convolution(layer, delay, transpose_kernel)
        elif get_type(layer) == 'DepthwiseConv2D':
            weights, biases = build_depthwise_convolution(layer, delay, transpose_kernel)
        self.set_biases(biases)

        exc_connections = [c for c in weights if c[2] > 0]
        inh_connections = [c for c in weights if c[2] < 0]
        

        if self.config.getboolean('tools', 'simulate'):
            self.connections.append(self.sim.Projection(
                self.layers[-2], self.layers[-1],
                (self.sim.FromListConnector(exc_connections,['weight', 'delay'])),
                receptor_type='excitatory',
                label=self.layers[-1].label +'_excitatory'))

            self.connections.append(self.sim.Projection(
                self.layers[-2], self.layers[-1],
                (self.sim.FromListConnector(inh_connections,['weight', 'delay'])),
                receptor_type='inhibitory',
                label=self.layers[-1].label +'_inhibitory'))
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
        self.sim.set_number_of_neurons_per_core(self.sim.IF_curr_exp, 128)
        data = kwargs[str('x_b_l')]
        if self.data_format == 'channels_last' and data.ndim == 4:
            data = np.moveaxis(data, 3, 1)
        
        x_flat = np.ravel(data)
        if self._poisson_input:
            rates = 1000 * x_flat / self.rescale_fac
            self.layers[0].set(rate=rates)
        elif self._dataset_format == 'aedat':
            raise NotImplementedError
        else:
            spike_times = \
                [np.linspace(0, self._duration, self._duration * amplitude)
                 for amplitude in x_flat]
            self.layers[0].set(spike_times=spike_times)

        from pynn_object_serialisation.functions import intercept_simulator
        import pylab
        current_time = pylab.datetime.datetime.now().strftime("_%H%M%S_%d%m%Y")
        intercept_simulator(self.sim, "snn_toolbox_spinnaker_" + current_time,
                            post_abort=False)
        self.sim.run(self._duration - self._dt)
        print("\nCollecting results...")
        output_b_l_t = self.get_recorded_vars(self.layers)

        return output_b_l_t
