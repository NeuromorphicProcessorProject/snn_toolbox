# -*- coding: utf-8 -*-
"""Building SNNs using Brian2.

The modules in ``target_simulators`` package allow building a spiking network
and exporting it for use in a spiking simulator.

This particular module offers functionality for Brian2 simulator. Adding
another simulator requires implementing the class ``SNN_compiled`` with its
methods tailored to the specific simulator.

Created on Thu May 19 15:00:02 2016

@author: rbodo
"""

# For compatibility with python2
from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

from random import randint

import numpy as np
from future import standard_library
from snntoolbox import echo
from snntoolbox.config import settings, initialize_simulator

standard_library.install_aliases()


class SNN:
    """
    Class to hold the compiled spiking neural network, ready for testing in a
    spiking simulator.

    Attributes
    ----------

    sim: Simulator
        Module containing utility functions of spiking simulator. Result of
        calling ``snntoolbox.config.initialize_simulator()``. For instance, if
        using Brian simulator, this initialization would be equivalent to
        ``import pyNN.brian as sim``.

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

    def __init__(self, s=None):
        """Init function."""

        if s is None:
            s = settings

        self.sim = initialize_simulator(s['simulator'])
        self.connections = []
        self.threshold = 'v > v_thresh'
        self.reset = 'v = v_reset'
        self.eqs = 'dv/dt = -v/tau_m : volt'
        self.layers = []
        self.spikemonitors = []
        self.statemonitors = []
        self.snn = None
        self.parsed_model = None

    # noinspection PyUnusedLocal
    def build(self, parsed_model, **kwargs):
        """Compile SNN to prepare for simulation with Brian2.

        Parameters
        ----------

        parsed_model: Keras model
            Parsed input model; result of applying
            ``model_lib.extract(input_model)`` to the ``input model``.
        """

        self.parsed_model = parsed_model

        echo('\n' + "Compiling spiking network...\n")

        self.add_input_layer(parsed_model.layers[0].batch_input_shape)

        # Iterate over hidden layers to create spiking neurons and store
        # connections.
        for layer in parsed_model.layers:
            layer_type = layer.__class__.__name__
            if 'Flatten' in layer_type:
                continue
            echo("Building layer: {}\n".format(layer.name))
            self.add_layer(layer)
            if layer_type == 'Dense':
                self.build_dense(layer)
            elif layer_type == 'Convolution2D':
                self.build_convolution(layer)
            elif layer_type in {'MaxPooling2D', 'AveragePooling2D'}:
                self.build_pooling(layer)

        echo("Compilation finished.\n\n")

        # Track the output layer spikes. Add monitor here if it was not already
        # appended above (because settings['verbose'] < 1)
        if len(self.spikemonitors) < len(self.layers):
            self.spikemonitors.append(self.sim.SpikeMonitor(self.layers[-1]))

        # Create snapshot of network
        self.store()

    def add_input_layer(self, input_shape):
        """Configure input layer."""

        self.layers.append(self.sim.PoissonGroup(
            np.prod(input_shape[1:]), rates=0*self.sim.Hz,
            dt=settings['dt']*self.sim.ms))
        self.spikemonitors.append(self.sim.SpikeMonitor(self.layers[0]))
        self.layers[0].add_attribute('label')
        self.layers[0].label = 'InputLayer'

    def add_layer(self, layer):
        """Add empty layer."""

        self.layers.append(self.sim.NeuronGroup(
            np.prod(layer.output_shape[1:]), model=self.eqs,
            threshold=self.threshold, reset=self.reset,
            dt=settings['dt']*self.sim.ms, method='linear'))
        self.connections.append(self.sim.Synapses(
            self.layers[-2], self.layers[-1], model='w:volt', on_pre='v+=w',
            dt=settings['dt']*self.sim.ms))
        self.layers[-1].add_attribute('label')
        self.layers[-1].label = layer.name
        if settings['verbose'] > 1:
            self.spikemonitors.append(self.sim.SpikeMonitor(self.layers[-1]))
        if settings['verbose'] == 3:
            self.statemonitors.append(self.sim.StateMonitor(self.layers[-1],
                                                            'v', record=True))

    def build_dense(self, layer):
        """Build dense layer."""

        weights = layer.get_weights()[0]  # [W, b][0]
        self.connections[-1].connect(True)
        self.connections[-1].w = weights.flatten() * self.sim.volt

    def build_convolution(self, layer):
        """Build convolution layer."""

        weights = layer.get_weights()[0]  # [W, b][0]
        nx = layer.input_shape[3]  # Width of feature map
        ny = layer.input_shape[2]  # Hight of feature map
        kx = layer.nb_col  # Width of kernel
        ky = layer.nb_row  # Hight of kernel
        px = int((kx - 1) / 2)  # Zero-padding columns
        py = int((ky - 1) / 2)  # Zero-padding rows
        if layer.border_mode == 'valid':
            # In border_mode 'valid', the original sidelength is
            # reduced by one less than the kernel size.
            mx = nx - kx + 1  # Number of columns in output filters
            my = ny - ky + 1  # Number of rows in output filters
            x0 = px
            y0 = py
        elif layer.border_mode == 'same':
            mx = nx
            my = ny
            x0 = 0
            y0 = 0
        else:
            raise Exception("Border_mode {} not supported".format(
                layer.border_mode))
        # Loop over output filters 'fout'
        for fout in range(weights.shape[0]):
            for y in range(y0, ny - y0):
                for x in range(x0, nx - x0):
                    target = x - x0 + (y - y0) * mx + fout * mx * my
                    # Loop over input filters 'fin'
                    for fin in range(weights.shape[1]):
                        for k in range(-py, py + 1):
                            if not 0 <= y + k < ny:
                                continue
                            source = x + (y + k) * nx + fin * nx * ny
                            for l in range(-px, px + 1):
                                if not 0 <= x + l < nx:
                                    continue
                                self.connections[-1].connect(i=source+l,
                                                             j=target)
                                self.connections[-1].w[source + l, target] = (
                                    weights[fout, fin, py-k, px-l] *
                                    self.sim.volt)
                echo('.')
            echo(' {:.1%}\n'.format(((fout + 1) * weights.shape[1]) /
                 (weights.shape[0] * weights.shape[1])))

    def build_pooling(self, layer):
        """Build pooling layer."""

        nx = layer.input_shape[3]  # Width of feature map
        ny = layer.input_shape[2]  # Hight of feature map
        dx = layer.pool_size[1]  # Width of pool
        dy = layer.pool_size[0]  # Hight of pool
        sx = layer.strides[1]
        sy = layer.strides[0]
        if layer.__class__.__name__ == 'MaxPooling2D':
            echo("WARNING: Layer type 'MaxPooling' not supported yet. " +
                 "Falling back on 'AveragePooling'.\n")
            for fout in range(layer.input_shape[1]):  # Feature maps
                for y in range(0, ny-dy+1, sy):
                    for x in range(0, nx-dx+1, sx):
                        target = int(x/sx+y/sy*((nx-dx)/sx+1) +
                                     fout*nx*ny/(dx * dy))
                        for k in range(dy):
                            source = x+(y+k)*nx+fout*nx * ny
                            for l in range(dx):
                                self.connections[-1].connect(i=source+l,
                                                             j=target)
                    echo('.')
                echo(' {:.1%}\n'.format((1 + fout) / layer.input_shape[1]))
            self.connections[-1].w = self.sim.volt / (dx * dy)
        elif layer.__class__.__name__ == 'AveragePooling2D':
            for fout in range(layer.input_shape[1]):  # Feature maps
                for y in range(0, ny-dy+1, sy):
                    for x in range(0, nx-dx+1, sx):
                        target = int(x/sx+y/sy*((nx-dx)/sx+1) +
                                     fout*nx*ny/(dx*dy))
                        for k in range(dy):
                            source = x+(y+k)*nx+fout*nx*ny
                            for l in range(dx):
                                self.connections[-1].connect(i=source+l,
                                                             j=target)
                    echo('.')
                echo(' {:.1%}\n'.format((1 + fout) / layer.input_shape[1]))
            self.connections[-1].w = self.sim.volt / (dx * dy)

    def store(self):
        """Store network by creating Network object."""

        self.snn = self.sim.Network(self.layers, self.connections,
                                    self.spikemonitors, self.statemonitors)

    def run(self, x_test, y_test, **kwargs):
        """Simulate a spiking network with IF units and Poisson input in pyNN.

        Simulate a spiking network with IF units and Poisson input in pyNN,
        using a simulator like Brian, NEST, NEURON, etc.

        This function will randomly select ``settings['num_to_test']`` test
        samples among ``x_test`` and simulate the network on those.

        Alternatively, a list of specific input samples can be given to the
        toolbox GUI, which will then be used for testing.

        If ``settings['verbose'] > 1``, the simulator records the
        spiketrains and membrane potential of each neuron in each layer, for
        the last sample.

        This is somewhat costly in terms of memory and time, but can be useful
        for debugging the network's general functioning.

        Parameters
        ----------

        x_test : float32 array
            The input samples to test. With data of the form
            (channels, num_rows, num_cols), x_test has dimension
            (num_samples, channels*num_rows*num_cols) for a multi-layer
            perceptron, and (num_samples, channels, num_rows, num_cols) for a
            convolutional net.
        y_test : float32 array
            Ground truth of test data. Has dimension (num_samples, num_classes)
        kwargs: Optional[dict]
            - s: Optional[dict]
                Settings. If not given, the ``snntoolobx.config.settings``
                dictionary is used.
            - path: Optional[str]
                Where to store the output plots. If no path given, this value is
                taken from the settings dictionary.

        Returns
        -------

        total_acc : float
            Number of correctly classified samples divided by total number of
            test samples.
        """

        from snntoolbox.io_utils.plotting import plot_confusion_matrix

        s = kwargs['settings'] if 'settings' in kwargs else settings
        log_dir = kwargs['path'] if 'path' in kwargs \
            else s['log_dir_of_current_run']

        # Load input layer
        input_layer = None
        for obj in self.snn.objects:
            if 'poissongroup' in obj.name and 'thresholder' not in obj.name:
                input_layer = obj
        assert input_layer, "No input layer found."

        # Update parameters
        namespace = {'v_thresh': s['v_thresh'] * self.sim.volt,
                     'v_reset': s['v_reset'] * self.sim.volt,
                     'tau_m': s['tau_m'] * self.sim.ms}
        results = []
        guesses = []
        truth = []

        # Iterate over the number of samples to test
        for test_num in range(s['num_to_test']):
            # If a list of specific input samples is given, iterate over that,
            # and otherwise pick a random test sample from among all possible
            # input samples in x_test.
            si = s['sample_indices_to_test']
            ind = randint(0, len(x_test) - 1) if si == [] else si[test_num]

            # Add Poisson input.
            if s['verbose'] > 1:
                echo("Creating poisson input...\n")
            input_layer.rates = x_test[ind, :].flatten() * s['input_rate'] * \
                self.sim.Hz

            # Run simulation for 'duration'.
            if s['verbose'] > 1:
                echo("Starting new simulation...\n")
            self.snn.store()
            self.snn.run(s['duration'] * self.sim.ms, namespace=namespace)

            # Get result by comparing the guessed class (i.e. the index of the
            # neuron in the last layer which spiked most) to the ground truth.
            guesses.append(np.argmax(self.spikemonitors[-1].count))
            truth.append(np.argmax(y_test[ind, :]))
            results.append(guesses[-1] == truth[-1])

            if s['verbose'] > 0:
                echo("Sample {} of {} completed.\n".format(test_num + 1,
                     s['num_to_test']))
                echo("Moving average accuracy: {:.2%}.\n".format(
                    np.mean(results)))

            if s['verbose'] > 1 and test_num == s['num_to_test'] - 1:
                echo("Simulation finished. Collecting results...\n")
                self.collect_plot_results(x_test[ind:ind+s['batch_size']],
                                          test_num)

            # Reset simulation time and recorded network variables for next
            # run.
            if s['verbose'] > 1:
                echo("Resetting simulator...\n")
            # Skip during last run so the recorded variables are not discarded
            if test_num < s['num_to_test'] - 1:
                self.snn.restore()
            if s['verbose'] > 1:
                echo("Done.\n")

        if s['verbose'] > 1:
            plot_confusion_matrix(truth, guesses, log_dir)

        total_acc = np.mean(results)
        ss = '' if s['num_to_test'] == 1 else 's'
        echo("Total accuracy: {:.2%} on {} test sample{}.\n\n".format(
             total_acc, s['num_to_test'], ss))

        self.snn.restore()

        return total_acc

    @staticmethod
    def end_sim():
        """Clean up after simulation."""

        pass

    @staticmethod
    def save(path=None, filename=None):
        """Write model architecture and parameters to disk."""

        pass

    def collect_plot_results(self, x_batch, idx=0):
        """Collect spiketrains of all ``layers`` of a net.

        Collect spiketrains of all ``layers`` of a net from one simulation run,
        and plot results.

        Plots include: Spiketrains, activations, spikerates, membrane
        potential, correlations.

        To visualize the spikerates, neurons in hidden layers are spatially
        arranged on a 2d rectangular grid, and the firing rate of each neuron
        on the grid is encoded by color.

        Membrane potential vs time is plotted for all except the input layer.

        The activations are obtained by evaluating the original ANN on a sample
        ``x_batch``. The optional integer ``idx`` represents the index of a
        specific sample to plot.
        """

        from snntoolbox.io_utils.plotting import output_graphs, plot_potential
        from snntoolbox.core.util import get_activations_batch

        # Collect spiketrains of all layers, for the last test sample.
        vmem = []
        show_legend = False

        # Allocate a list 'spiketrains_batch' with the following specification:
        # Each entry in ``spiketrains_batch`` contains a tuple
        # ``(spiketimes, label)`` for each layer of the network (for the first
        # batch only, and excluding ``Flatten`` layers).
        # ``spiketimes`` is an array where the last index contains the spike
        # times of the specific neuron, and the first indices run over the
        # number of neurons in the layer:
        # (num_to_test, n_chnls*n_rows*n_cols, duration)
        # ``label`` is a string specifying both the layer type and the index,
        # e.g. ``'03Dense'``.
        spiketrains_batch = []
        j = 0
        for i, layer in enumerate(self.layers[1:]):
            # Skip Flatten layer (only present in ``parsed_model``)
            if 'Flatten' in self.parsed_model.layers[i].__class__.__name__:
                j += 1
            shape = list(self.parsed_model.layers[i+j].output_shape) + \
                [int(settings['duration'] / settings['dt'])]
            shape[0] = 1  # simparams['num_to_test']
            spiketrains_batch.append((np.zeros(shape), layer.label))
            spiketrain_dict = self.spikemonitors[i].spike_trains()
            spiketrains = np.array(
                [spiketrain_dict[key] / self.sim.ms for key in
                 spiketrain_dict.keys()])
            spiketrains_full = np.empty((np.prod(shape[:-1]), shape[-1]))
            for k in range(len(spiketrains_full)):
                spiketrain = np.zeros(shape[-1])
                spiketrain[:len(spiketrains[k])] = np.array(
                    spiketrains[k][:shape[-1]])
                spiketrains_full[k] = spiketrain
            spiketrains_batch[i][0][:] = np.reshape(spiketrains_full, shape)
            # Repeat for membrane potential
            if settings['verbose'] == 3:
                vm = [np.array(v/1e6/self.sim.mV).transpose() for v in
                      self.statemonitors[i-1].v]
                vmem.append((vm, layer.label))
                times = self.statemonitors[0].t / self.sim.ms
                if i == len(self.layers) - 2:
                    show_legend = True
                plot_potential(times, vmem[-1], show_legend,
                               settings['log_dir_of_current_run'])

        activations_batch = get_activations_batch(self.parsed_model, x_batch)
        output_graphs(spiketrains_batch, activations_batch,
                      settings['log_dir_of_current_run'], idx)
