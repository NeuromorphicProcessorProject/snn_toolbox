# -*- coding: utf-8 -*-
"""
Created on Thu May 19 15:00:02 2016

@author: rbodo
"""

# For compatibility with python2
from __future__ import print_function, unicode_literals
from __future__ import division, absolute_import
from future import standard_library
from builtins import int, range

import warnings
import numpy as np
from random import randint

from snntoolbox import echo
from snntoolbox.config import settings, initialize_simulator

standard_library.install_aliases()


class SNN_compiled():
    def __init__(self, ann):
        self.ann = ann
        self.sim = initialize_simulator(settings['simulator'])
        self.layers = [self.sim.PoissonGroup(np.prod(ann['input_shape'][1:]),
                                             rates=0*self.sim.Hz,
                                             dt=settings['dt']*self.sim.ms)]
        self.threshold = 'v > v_thresh'
        self.reset = 'v = v_reset'
        self.eqs = 'dv/dt = -v/tau_m : volt'
        self.spikemonitors = [self.sim.SpikeMonitor(self.layers[0])]
        self.statemonitors = []
        self.labels = ['InputLayer']

    def add_layer(self, layer):
        echo("Building layer: {}\n".format(layer['label']))
        self.labels.append(layer['label'])
        self.layers.append(self.sim.NeuronGroup(
            np.prod(layer['output_shape'][1:]), model=self.eqs,
            threshold=self.threshold, reset=self.reset,
            dt=settings['dt']*self.sim.ms, method='linear'))
        self.connections.append(self.sim.Synapses(
            self.layers[-2], self.layers[-1], model='w:volt', on_pre='v+=w',
            dt=settings['dt']*self.sim.ms))
        if settings['verbose'] > 1:
            self.spikemonitors.append(self.sim.SpikeMonitor(self.layers[-1]))
        if settings['verbose'] == 3:
            self.statemonitors.append(self.sim.StateMonitor(self.layers[-1],
                                                            'v', record=True))

    def build_dense(self, layer):
        weights = layer['weights'][0]  # [W, b][0]
        self.connections[-1].connect(True)
        self.connections[-1].w = weights.flatten() * self.sim.volt

    def build_convolution(self, layer):
        weights = layer['weights'][0]  # [W, b][0]
        nx = layer['input_shape'][3]  # Width of feature map
        ny = layer['input_shape'][2]  # Hight of feature map
        kx = layer['nb_col']  # Width of kernel
        ky = layer['nb_row']  # Hight of kernel
        px = int((kx - 1) / 2)  # Zero-padding columns
        py = int((ky - 1) / 2)  # Zero-padding rows
        if layer['border_mode'] == 'valid':
            # In border_mode 'valid', the original sidelength is
            # reduced by one less than the kernel size.
            mx = nx - kx + 1  # Number of columns in output filters
            my = ny - ky + 1  # Number of rows in output filters
            x0 = px
            y0 = py
        elif layer['border_mode'] == 'same':
            mx = nx
            my = ny
            x0 = 0
            y0 = 0
        else:
            raise Exception("Border_mode {} not supported".format(
                layer['border_mode']))
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
        if layer['layer_type'] == 'MaxPooling2D':
            echo("WARNING: Layer type 'MaxPooling' not supported yet. " +
                 "Falling back on 'AveragePooling'.\n")
        nx = layer['input_shape'][3]  # Width of feature map
        ny = layer['input_shape'][2]  # Hight of feature map
        dx = layer['pool_size'][1]  # Width of pool
        dy = layer['pool_size'][0]  # Hight of pool
        sx = layer['strides'][1]
        sy = layer['strides'][0]
        for fout in range(layer['input_shape'][1]):  # Feature maps
            for y in range(0, ny - dy + 1, sy):
                for x in range(0, nx - dx + 1, sx):
                    target = int(x / sx + y / sy * ((nx - dx) / sx + 1) +
                                 fout * nx * ny / (dx * dy))
                    for k in range(dy):
                        source = x + (y + k) * nx + fout * nx * ny
                        for l in range(dx):
                            self.connections[-1].connect(i=source+l, j=target)
                echo('.')
            echo(' {:.1%}\n'.format((1 + fout) / layer['input_shape'][1]))
        self.connections[-1].w = self.sim.volt / (dx * dy)

    def store(self):
        self.snn = self.sim.Network(self.layers, self.connections,
                                    self.spikemonitors, self.statemonitors)

    def build(self):
        """
        Convert an `analog` to a `spiking` neural network.

        Written in pyNN (http://neuralensemble.org/docs/PyNN/).
        pyNN is a simulator-independent language for building neural network
        models. It allows running the converted net in a Spiking Simulator like
        Brian, NEURON, or NEST.

        During conversion, two lists are created and stored to disk: ``layers``
        and ``connections``. Each entry in ``layers`` represents a population
        of neurons, given by a pyNN ``Population`` object. The neurons in these
        layers are connected by pyNN ``Projection`` s, stored in
        ``connections`` list.
        This conversion method performs the connection process between layers.
        This means, if the session was started with a call to ``sim.setup()``,
        the converted network can be tested right away, using the simulator
        ``sim``.
        However, when starting a new session (calling ``sim.setup()`` after
        conversion), the ``layers`` have to be reloaded from disk using
        ``io.load.load_assembly``, and the connections reestablished
        manually. This is implemented in ``simulation.run_SNN``, go there for
        details. See ``tests.util.test_full`` about how to simulate after
        converting. The script ``tests.parameter_sweep`` wraps all these
        functions in one and allows toggling them by setting a few parameters.

        Parameters
        ----------
        ann : ANN model
            The network architecture and weights as ``snntoolbox.io.load.ANN``
            object.

        Returns
        -------
        layers : list
            Each entry represents a layer, i.e. a population of neurons, in
            form of pyNN ``Population`` objects.
        """

        echo('\n')
        echo("Compiling spiking network...\n")

        # Iterate over hidden layers to create spiking neurons and store
        # connections.
        for layer in self.ann['layers']:
            if layer['layer_type'] in {'Dense', 'Convolution2D',
                                       'MaxPooling2D', 'AveragePooling2D'}:
                self.add_layer(layer)
            else:
                echo("Skipped layer:  {}\n".format(layer['layer_type']))
                continue
            if layer['layer_type'] == 'Dense':
                self.build_dense(layer)
            elif layer['layer_type'] == 'Convolution2D':
                self.build_convolution(layer)
            elif layer['layer_type'] in {'MaxPooling2D', 'AveragePooling2D'}:
                self.build_pooling(layer)

        echo("Compilation finished.\n\n")

        # Track the output layer spikes. Add monitor here if it was not already
        # appended above (because globalparams['verbose'] < 1)
        if len(self.spikemonitors) < len(self.layers):
            self.spikemonitors.append(self.sim.SpikeMonitor(self.layers[-1]))

        # Create snapshot of network
        self.store()

    def run(self, net, X_test, Y_test):
        """
        Simulate a spiking network with IF units and Poisson input in pyNN,
        using a simulator like Brian, NEST, NEURON, etc.

        This function will randomly select ``simparams['num_to_test']`` test
        samples among ``X_test`` and simulate the network on those.

        If ``globalparams['verbose'] > 1``, the simulator records the
        spiketrains and membrane potential of each neuron in each layer.
        Doing so for all ``simparams['num_to_test']`` test samples is very
        costly in terms of memory and time, but can be useful for debugging the
        network's general functioning. This function returns only the
        recordings of the last test sample. To get detailed information about
        the network's behavior for a particular sample, replace the test set
        ``X_test, Y_test`` with this sample of interest.

        Parameters
        ----------

        X_test : float32 array
            The input samples to test.
            With data of the form (channels, num_rows, num_cols),
            X_test has dimension (num_samples, channels*num_rows*num_cols)
            for a multi-layer perceptron, and
            (num_samples, channels, num_rows, num_cols)
            for a convolutional net.
        Y_test : float32 array
            Ground truth of test data. Has dimension
            (num_samples, num_classes).
        layers : list, possible kwarg
            Each entry represents a layer, i.e. a population of neurons, in
            form of pyNN ``Population`` objects.

        Returns
        -------

        total_acc : float
            Number of correctly classified samples divided by total number of
            test samples.
        spiketrains : list of tuples
            Each entry in ``spiketrains`` contains a tuple
            ``(spiketimes, label)`` for each layer of the network
            (for the last test sample only).
            ``spiketimes`` is a 2D array where the first index runs over the
            number of neurons in the layer, and the second index contains the
            spike times of the specific neuron.
            ``label`` is a string specifying both the layer type and the index,
            e.g. ``'03Dense'``.
        vmem : list of tuples
            Each entry in ``vmem`` contains a tuple ``(vm, label)`` for each
            layer of the network (for the first test sample only). ``vm`` is a
            2D array where the first index runs over the number of neurons in
            the layer, and the second index contains the membrane voltage of
            the specific neuron.
            ``label`` is a string specifying both the layer type and the index,
            e.g. ``'03Dense'``.
        """

        for obj in self.snn.objects:
            if 'poissongroup' in obj.name and 'thresholder' not in obj.name:
                input_layer = obj
        namespace = {'v_thresh': settings['v_thresh'] * self.sim.volt,
                     'v_reset': settings['v_reset'] * self.sim.volt,
                     'tau_m': settings['tau_m'] * self.sim.ms}
        results = []

        # Iterate over the number of samples to test
        for test_num in range(settings['num_to_test']):
            # If a list of specific input samples is given, iterate over that,
            # and otherwise pick a random test sample from among all possible
            # input samples in X_test.
            si = settings['sample_indices_to_test']
            ind = randint(0, len(X_test) - 1) if si == [] else si[test_num]

            # Add Poisson input.
            if settings['verbose'] > 1:
                echo("Creating poisson input...\n")
            input_layer.rates = X_test[ind, :].flatten() * \
                settings['max_f'] * self.sim.Hz

            # Run simulation for 'duration'.
            if settings['verbose'] > 1:
                echo("Starting new simulation...\n")
            self.snn.store()
            self.snn.run(settings['duration'] * self.sim.ms,
                         namespace=namespace)

            # Get result by comparing the guessed class (i.e. the index of the
            # neuron in the last layer which spiked most) to the ground truth.
            guesses = np.argmax(self.spikemonitors[-1].count)
            truth = np.argmax(Y_test[ind, :])
            results.append(guesses == truth)
            if settings['verbose'] > 0:
                echo("Sample {} of {} completed.\n".format(test_num + 1,
                     settings['num_to_test']))
                echo("Moving average accuracy: {:.2%}.\n".format(
                    np.mean(results)))

            if settings['verbose'] > 1 and \
                    test_num == settings['num_to_test'] - 1:
                echo("Simulation finished. Collecting results...\n")
                self.collect_plot_results(net, X_test[ind])

            # Reset simulation time and recorded network variables for next
            # run.
            if settings['verbose'] > 1:
                echo("Resetting simulator...\n")
            # Skip during last run so the recorded variables are not discarded
            if test_num < settings['num_to_test'] - 1:
                self.snn.restore()
            if settings['verbose'] > 1:
                echo("Done.\n")

        total_acc = np.mean(results)
        s = '' if settings['num_to_test'] == 1 else 's'
        echo("Total accuracy: {:.2%} on {} test sample{}.\n\n".format(
             total_acc, settings['num_to_test'], s))

        self.snn.restore()

        return total_acc

    def end_sim(self):
        pass

    def collect_plot_results(self, net, X):
        # Plot spikerates and spiketrains of layers. To visualize the
        # spikerates, neurons in hidden layers are spatially arranged on a 2d
        # rectangular grid, and the firing rate of each neuron on the grid is
        # encoded by color. Also plot the membrane potential vs time (except
        # for the input layer).

        from snntoolbox.io_utils.plotting import output_graphs, plot_potential
        # Turn off warning because we have no influence on it:
        # "FutureWarning: elementwise comparison failed; returning scalar
        #  instead, but in the future will perform elementwise comparison"
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            warnings.warn('deprecated', FutureWarning)

        # Allocate a list 'spiketrains' with the following specification:
        # Each entry in ``spiketrains`` contains a tuple
        # ``(spiketimes, label)`` for each layer of the network (for the
        # first batch only, and excluding ``Flatten`` layers).
        # ``spiketimes`` is an array where the first indices run over the
        # number of neurons in the layer, and the last index contains the
        # spike times of the specific neuron.
        # ``label`` is a string specifying both the layer type and the
        # index, e.g. ``'03Dense'``.
        spiketrains_batch = []
        for (i, layer) in enumerate(net.layers):
            if 'get_activ' in layer.keys():
                shape = list(layer['output_shape']) + \
                             [int(settings['duration'] / settings['dt'])]
                shape[0] = 1  # simparams['num_to_test']
                idx = i if 'Pool' in layer['label'] else i-1
                spiketrains_batch.append((np.zeros(shape),
                                          net.layers[idx]['label']))

            # Collect spiketrains of all layers, for the last test sample.
            for i in range(len(self.labels)-1):
                shape = spiketrains_batch[i][0].shape
                spiketrains = np.array(
                    self.spikemonitors[i+1].spike_trains() / self.sim.ms)
                spiketrains2 = np.empty((np.prod(shape[:-1]), shape[-1]))
                for j in range(len(spiketrains)):
                    spiketrain = np.zeros(shape[-1])
                    spiketrain[:len(spiketrains[j])] = np.array(spiketrains[j])
                    spiketrains2[j] = spiketrain
                spiketrains_batch[i][0][:] = np.reshape(spiketrains2, shape)
            output_graphs(spiketrains_batch, net, np.array(X, ndmin=4),
                          settings['log_dir_of_current_run'])
            # Maybe repeat for membrane potential, skipping input layer
            vmem = []
            if settings['verbose'] == 3:
                showLegend = False
                for i in range(len(self.labels)-1):
                    vm = []
                    for v in self.statemonitors[i].v:
                        vm.append(np.array(v/1e6/self.sim.mV).transpose())
                    vmem.append((vm, self.labels[i+1]))
                    times = self.statemonitors[0].t / self.sim.ms
                    if settings['verbose'] == 3:
                        if i == len(self.labels) - 2:
                            showLegend = True
                        plot_potential(times, vmem[i], showLegend=showLegend)
