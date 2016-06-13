# -*- coding: utf-8 -*-
"""

The modules in ``target_simulators`` package allow building a spiking network
and exporting it for use in a spiking simulator.

This particular module offers functionality for pyNN simulators Brian, Nest,
Neuron. Adding another simulator requires implementing the class
``SNN_compiled`` with its methods tailored to the specific simulator.

Created on Thu May 19 15:00:02 2016

@author: rbodo
"""

# For compatibility with python2
from __future__ import print_function, unicode_literals
from __future__ import division, absolute_import
from future import standard_library
from builtins import int, range

import os
import sys
import numpy as np
from random import randint
from six.moves import cPickle

# Turn off warning because we have no influence on it:
# "UserWarning: ConvergentConnect is deprecated and will be removed
# in a future version of NEST. Please use Connect instead!"
import warnings

from snntoolbox import echo
from snntoolbox.config import settings, initialize_simulator
from snntoolbox.io_utils.save import confirm_overwrite

standard_library.install_aliases()

cellparams_pyNN = {'v_thresh', 'v_reset', 'v_rest', 'e_rev_E', 'e_rev_I', 'cm',
                   'i_offset', 'tau_refrac', 'tau_m', 'tau_syn_E', 'tau_syn_I'}


class SNN_compiled():
    """
    Class to hold the compiled spiking neural network, ready for testing in a
    spiking simulator.

    Parameters
    ----------

    ann: dict
        Parsed input model; result of applying ``model_lib.extract(in)`` to the
        input model ``in``.

    Attributes
    ----------

    ann: dict
        Parsed input model; result of applying ``model_lib.extract(in)`` to the
        input model ``in``.

    sim: Simulator
        Module containing utility functions of spiking simulator. Result of
        calling ``snntoolbox.config.initialize_simulator()``. For instance, if
        using Brian simulator, this initialization would be equivalent to
        ``import pyNN.brian as sim``.

    layers: list
        Each entry represents a layer, i.e. a population of neurons, in form of
        pyNN ``Population`` objects.

    connections: list
        pyNN ``Projection`` objects representing the connections between
        individual layers.

    labels: list
        The layer labels.

    output_shapes: list
        The output shapes of each layer. During conversion, all layers are
        flattened. Need output shapes to reshape the output of layers back to
        original form when plotting results later.

    cellparams: dict
        Neuron cell parameters determining properties of the spiking neurons in
        pyNN simulators.

    Methods
    -------

        build:
            Convert an ANN to a spiking neural network, using layers derived
            from Keras base classes.
        run:
            Simulate a spiking network.
        save:
            Write model architecture and weights to disk.
        load:
            Load model architecture and weights from disk.
        end_sim:
            Clean up after simulation.

    """

    def __init__(self, ann):
        self.ann = ann
        self.sim = initialize_simulator()
        self.layers = [self.sim.Population(
            int(np.prod(ann['input_shape'][1:])),
            self.sim.SpikeSourcePoisson(), label='InputLayer')]
        self.connections = []
        self.labels = []
        self.output_shapes = []
        self.cellparams = {key: settings[key] for key in cellparams_pyNN}

    def build(self):
        """
        Compile a spiking neural network to prepare for simulation.

        Written in pyNN (http://neuralensemble.org/docs/PyNN/).
        pyNN is a simulator-independent language for building neural network
        models. It allows running the converted net in a Spiking Simulator like
        Brian, NEURON, or NEST.

        During compilation, two lists are created and stored to disk:
        ``layers`` and ``connections``. Each entry in ``layers`` represents a
        population of neurons, given by a pyNN ``Population`` object. The
        neurons in these layers are connected by pyNN ``Projection`` s, stored
        in ``connections`` list.

        This compilation method performs the connection process between layers.
        This means, if the session was started with a call to ``sim.setup()``,
        the converted network can be tested right away, using the simulator
        ``sim``.

        However, when starting a new session (calling ``sim.setup()`` after
        conversion), the ``layers`` have to be reloaded from disk using
        ``load_assembly``, and the connections reestablished manually. This is
        implemented in ``run`` method, go there for details.
        See ``snntoolbox.core.pipeline.test_full`` about how to simulate after
        converting.

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
            self.connect_layer()

        echo("Compilation finished.\n\n")

    def add_layer(self, layer):
        echo("Building layer: {}\n".format(layer['label']))
        self.conns = []
        self.labels.append(layer['label'])
        self.layers.append(self.sim.Population(
            int(np.prod(layer['output_shape'][1:])), self.sim.IF_cond_exp,
            self.cellparams, label=layer['label']))
        self.output_shapes.append(layer['output_shape'])
        if 'activation' in layer and layer['activation'] == 'softmax':
            echo("WARNING: Activation 'softmax' not implemented. " +
                 "Using 'relu' activation instead.\n")

    def build_dense(self, layer):
        weights = layer['weights'][0]  # [W, b][0]
        for i in range(len(weights)):
            for j in range(len(weights[0])):
                self.conns.append((i, j, weights[i, j], settings['delay']))

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
                                self.conns.append((source + l, target,
                                                   weights[fout, fin,
                                                           py-k, px-l],
                                                   settings['delay']))
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
                            self.conns.append((source + l, target,
                                               1 / (dx * dy),
                                               settings['delay']))
                echo('.')
            echo(' {:.1%}\n'.format((1 + fout) / layer['input_shape'][1]))

    def connect_layer(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            warnings.warn('deprecated', UserWarning)

            self.connections.append(self.sim.Projection(
                self.layers[-2], self.layers[-1],
                self.sim.FromListConnector(self.conns, ['weight', 'delay'])))

    def run(self, snn_precomp, X_test, Y_test):
        """
        Simulate a spiking network with IF units and Poisson input in pyNN,
        using a simulator like Brian, NEST, NEURON, etc.

        This function will randomly select ``settings['num_to_test']`` test
        samples among ``X_test`` and simulate the network on those.

        Alternatively, a list of specific input samples can be given to the
        toolbox GUI, which will then be used for testing.

        If ``settings['verbose'] > 1``, the simulator records the
        spiketrains and membrane potential of each neuron in each layer, for
        the last sample.

        This is somewhat costly in terms of memory and time, but can be useful
        for debugging the network's general functioning.

        Parameters
        ----------

        snn_precomp: SNN
            The converted spiking network, before compilation (i.e. independent
            of simulator).
        X_test: float32 array
            The input samples to test. With data of the form
            (channels, num_rows, num_cols), X_test has dimension
            (num_samples, channels*num_rows*num_cols) for a multi-layer
            perceptron, and (num_samples, channels, num_rows, num_cols) for a
            convolutional net.
        Y_test: float32 array
            Ground truth of test data. Has dimension (num_samples, num_classes)

        Returns
        -------

        total_acc: float
            Number of correctly classified samples divided by total number of
            test samples.

        """

        from snntoolbox.io_utils.plotting import plot_confusion_matrix

        # Setup pyNN simulator if it was not passed on from a previous session.
        if len(self.layers) == 1:  # Contains only input layer
            self.load(settings['path'], settings['filename_snn_exported'])

        # Set cellparameters of neurons in each layer and initialize membrane
        # potential.
        for layer in self.layers[1:]:
            layer.set(**self.cellparams)
            layer.initialize(v=self.layers[1].get('v_rest'))
        # The spikes of the last layer are recorded by default because they
        # contain the networks output (classification guess).
        self.layers[-1].record(['spikes'])

        results = []
        guesses = []
        truth = []

        # Iterate over the number of samples to test
        for test_num in range(settings['num_to_test']):
            # Specify variables to record. For performance reasons, record
            # spikes and potential only for the last test sample. Have to
            # reload network in order to tell the layers to record new
            # variables.
            if settings['verbose'] > 1 and \
                    test_num == settings['num_to_test'] - 1:
                if settings['num_to_test'] > 1:
                    echo("For last run, record spike rates and membrane " +
                         "potential of all layers.\n")
                    self.load()
                self.layers[0].record(['spikes'])
                for layer in self.layers[1:]:
                    layer.set(**self.cellparams)
                    layer.initialize(v=self.layers[1].get('v_rest'))
                    if settings['verbose'] == 3:
                        layer.record(['spikes', 'v'])
                    else:
                        layer.record(['spikes'])

            # If a list of specific input samples is given, iterate over that,
            # and otherwise pick a random test sample from among all possible
            # input samples in X_test.
            si = settings['sample_indices_to_test']
            ind = randint(0, len(X_test) - 1) if si == [] else si[test_num]

            # Add Poisson input.
            if settings['verbose'] > 1:
                echo("Creating poisson input...\n")
            rates = X_test[ind, :].flatten()
            for (i, ss) in enumerate(self.layers[0]):
                ss.rate = rates[i] * settings['input_rate']

            # Run simulation for 'duration'.
            if settings['verbose'] > 1:
                echo("Starting new simulation...\n")
            self.sim.run(settings['duration'])

            # Get result by comparing the guessed class (i.e. the index of the
            # neuron in the last layer which spiked most) to the ground truth.
            output = [len(spiketrain) for spiketrain in
                      self.layers[-1].get_data().segments[-1].spiketrains]
            guesses.append(np.argmax(output))
            truth.append(np.argmax(Y_test[ind, :]))
            results.append(guesses[-1] == truth[-1])

            if settings['verbose'] > 0:
                echo("Sample {} of {} completed.\n".format(test_num + 1,
                     settings['num_to_test']))
                echo("Moving average accuracy: {:.2%}.\n".format(
                    np.mean(results)))

            if settings['verbose'] > 1 and \
                    test_num == settings['num_to_test'] - 1:
                echo("Simulation finished. Collecting results...\n")
                collect_plot_results(self.layers, self.output_shapes,
                                     snn_precomp,
                                     X_test[ind:ind+settings['batch_size']])

            # Reset simulation time and recorded network variables for next run
            if settings['verbose'] > 1:
                echo("Resetting simulator...\n")
            self.sim.reset()
            if settings['verbose'] > 1:
                echo("Done.\n")

        if settings['verbose'] > 1:
            plot_confusion_matrix(truth, guesses,
                                  settings['log_dir_of_current_run'])

        total_acc = np.mean(results)
        s = '' if settings['num_to_test'] == 1 else 's'
        echo("Total accuracy: {:.2%} on {} test sample{}.\n\n".format(
             total_acc, settings['num_to_test'], s))

        return total_acc

    def end_sim(self):
        """ Clean up after simulation. """
        self.sim.end()

    def save(self, path=None, filename=None):
        """
        Write model architecture and weights to disk.

        Parameters
        ----------

        path: string, optional
            Path to directory where to save model. Defaults to
            ``settings['path']``.

        filename: string, optional
            Name of file to write model to. Defaults to
            ``settings['filename_snn_exported']``.
        """

        if path is None:
            path = settings['path']
        if filename is None:
            filename = settings['filename_snn_exported']

        self.save_assembly(path, filename)
        self.save_connections(path)

    def save_assembly(self, path=None, filename=None):
        """
        Write layers of neural network to disk.

        The size, structure, labels of all the population of an assembly are
        stored in a dictionary such that one can load them again using the
        ``snntoolbox.io.load.load_assembly`` function.

        The term "assembly" refers to pyNN internal nomenclature, where
        ``Assembly`` is a collection of layers (``Populations``), which in turn
        consist of a number of neurons (``cells``).

        Parameters
        ----------

        path: string, optional
            Path to directory where to save layers. Defaults to
            ``settings['path']``.

        filename: string, optional
            Name of file to write layers to. Defaults to
            ``settings['filename_snn_exported']``.

        """

        if path is None:
            path = settings['path']
        if filename is None:
            filename = settings['filename_snn_exported']

        filepath = os.path.join(path, filename)

        if not confirm_overwrite(filepath):
            return

        print("Saving assembly to {}...".format(filepath))

        s = {}
        labels = []
        variables = ['size', 'structure', 'label']
        for population in self.layers:
            labels.append(population.label)
            data = {}
            for variable in variables:
                data[variable] = getattr(population, variable)
            data['celltype'] = population.celltype.describe()
            s[population.label] = data
        s['labels'] = labels  # List of population labels describing the net.
        s['variables'] = variables  # List of variable names.
        s['size'] = len(self.layers)  # Number of populations in assembly.
        cPickle.dump(s, open(filepath, 'wb'))
        print("Done.\n")

    def save_connections(self, path=None):
        """
        Write weights of a neural network to disk.

        The weights between two layers are saved in a text file.
        They can then be used to connect pyNN populations e.g. with
        ``sim.Projection(layer1, layer2, sim.FromListConnector(filename))``,
        where ``sim`` is a simulator supported by pyNN, e.g. Brian, NEURON, or
        NEST.

        Parameters
        ----------

        path: string, optional
            Path to directory where connections are saved. Defaults to
            ``settings['path']``.

        Return
        ------
            Text files containing the layer connections. Each file is named
            after the layer it connects to, e.g. ``layer2.txt`` if connecting
            layer1 to layer2.

        """

        if path is None:
            path = settings['path']

        echo("Saving connections to {}...\n".format(path))

        # Iterate over layers to save each projection in a separate txt file.
        for projection in self.connections:
            filepath = os.path.join(path, projection.label.partition('→')[-1])
            if confirm_overwrite(filepath):
                projection.save('connections', filepath)
        echo("Done.\n")

    def load_assembly(self, path=None, filename=None):
        """
        Loads the populations in an assembly that was saved with the
        ``save_assembly`` function.

        The term "assembly" refers to pyNN internal nomenclature, where
        ``Assembly`` is a collection of layers (``Populations``), which in turn
        consist of a number of neurons (``cells``).

        Parameters
        ----------

        path: string, optional
            Path to directory where to load model from. Defaults to
            ``settings['path']``.

        filename: string, optional
            Name of file to load model from. Defaults to
            ``settings['filename_snn_exported']``.

        Returns
        -------

        layers: list
            List of pyNN ``Population`` objects.
        """

        if path is None:
            path = settings['path']
        if filename is None:
            filename = settings['filename_snn_exported']

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
            layers.append(population)

        if settings['verbose'] > 1:
            echo("Loaded spiking neuron layers from {}.\n".format(filepath))

        return layers

    def load(self, path=None, filename=None):
        """
        Load model architecture and weights from disk.

        Sets the ``snn`` and ``get_output`` attributes of this class.

        Parameters
        ----------

        path: string, optional
            Path to directory where to load model from. Defaults to
            ``settings['path']``.

        filename: string, optional
            Name of file to load model from. Defaults to
            ``settings['filename_snn_exported']``.

        """

        if path is None:
            path = settings['path']
        if filename is None:
            filename = settings['filename_snn_exported']

        self.layers = self.load_assembly(path, filename)
        for i in range(len(self.ann['layers'])):
            if 'get_activ' in self.ann['layers'][i].keys():
                idx = i if 'Pool' in self.ann['labels'][i] else i-1
                self.output_shapes.append(
                    self.ann['layers'][idx]['output_shape'])
        if settings['verbose'] > 1:
            echo("Restoring layer connections...\n")
        for i in range(len(self.layers)-1):
            filepath = os.path.join(path, self.layers[i+1].label)
            assert os.path.isfile(filepath), \
                "Connections were not found at specified location.\n"
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                warnings.warn('deprecated', UserWarning)
                self.sim.Projection(self.layers[i], self.layers[i+1],
                                    self.sim.FromFileConnector(filepath))


def collect_plot_results(layers, output_shapes, ann, X_batch, idx=0):
    """
    Collect spiketrains of all ``layers`` of a net from one simulation run, and
    plot results.

    Plots include: Spiketrains, activations, spikerates, membrane potential,
    correlations.

    To visualize the spikerates, neurons in hidden layers are spatially
    arranged on a 2d rectangular grid, and the firing rate of each neuron on
    the grid is encoded by color.

    Membrane potential vs time is plotted for all except the input layer.

    The activations are obtained by evaluating the original ANN ``ann`` on a
    batch of samples ``X_batch``. The optional integer ``idx`` represents the
    index of a specific sample to plot.

    The ``output shapes`` of each layer are needed to reshape the output of
    layers back to original form when plotting results (During conversion, all
    layers are flattened).

    """

    from snntoolbox.io_utils.plotting import output_graphs, plot_potential

    # Collect spiketrains of all layers, for the last test sample.
    vmem = []
    showLegend = False

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
    for (i, layer) in enumerate(layers):
        if i == 0 or 'Flatten' in layer.label:
            continue
        shape = list(output_shapes[j]) + \
            [int(settings['duration'] / settings['dt'])]
        shape[0] = 1  # simparams['num_to_test']
        spiketrains_batch.append((np.zeros(shape), layer.label))
        spiketrains = np.array(layer.get_data().segments[-1].spiketrains)
        spiketrains_full = np.empty((np.prod(shape[:-1]), shape[-1]))
        for k in range(len(spiketrains)):
            spiketrain = np.zeros(shape[-1])
            spiketrain[:len(spiketrains[k])] = np.array(
                spiketrains[k][:shape[-1]])
            spiketrains_full[k] = spiketrain
        spiketrains_batch[j][0][:] = np.reshape(spiketrains_full, shape)
        # Maybe repeat for membrane potential, skipping input layer
        if settings['verbose'] == 3 and i > 0:
            vm = [np.array(v) for v in
                  layer.get_data().segments[-1].analogsignalarrays]
            vmem.append((vm, layer.label))
            times = settings['dt'] * np.arange(len(vmem[0][0][0]))
            if i == len(layers) - 2:
                showLegend = True
            plot_potential(times, vmem[-1], showLegend=showLegend)
        j += 1

    output_graphs(spiketrains_batch, ann, X_batch,
                  settings['log_dir_of_current_run'], idx)
