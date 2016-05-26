# -*- coding: utf-8 -*-
"""
Created on Thu May 19 14:59:30 2016

@author: rbodo
"""

# For compatibility with python2
from __future__ import print_function, unicode_literals
from __future__ import division, absolute_import
from future import standard_library
from builtins import int, range

import os
import theano
from keras.models import Sequential
from snntoolbox import echo
from snntoolbox.config import settings, initialize_simulator
from snntoolbox.io_utils.plotting import output_graphs

standard_library.install_aliases()

__short_name__ = 'inisim'


class SNN_compiled():
    def __init__(self, ann):
        self.ann = ann
        self.sim = initialize_simulator(settings['simulator'])
        self.snn = Sequential()
        self.get_output = None

    def build(self):
        """
        Convert an ANN to a spiking neural network, using layers derived from
        Keras base classes.

        Aims at simulating the network on a self-implemented Integrate-and-Fire
        simulator using mean pooling and a timestepped approach.

        Parameters
        ----------
        ann : Keras model
            The network architecture and weights in json format.

        Returns
        -------
        snn : Keras model (derived class of Sequential)
            The spiking model
        get_output : Theano function
            Computes the output of the network
        """

        # Iterate over layers to create spiking neurons and connections.
        if settings['verbose'] > 1:
            echo("Iterating over ANN layers to add spiking layers...\n")
        for (layer_num, layer) in enumerate(self.ann['layers']):
            kwargs = {'name': layer['label'], 'trainable': False}
            kwargs2 = {}
            if layer_num == 0:
                # For the input layer, pass extra keyword argument
                # 'batch_input_shape' to layer constructor.
                input_shape = list(self.ann['input_shape'])
                input_shape[0] = settings['batch_size']
                input_time = theano.tensor.scalar('time')
                kwargs.update({'batch_input_shape': input_shape})
                kwargs2.update({'time_var': input_time})
            echo("Layer: {}\n".format(layer['label']))
            if layer['layer_type'] == 'Convolution2D':
                self.snn.add(self.sim.SpikeConv2DReLU(
                    layer['nb_filter'], layer['nb_row'], layer['nb_col'],
                    self.sim.floatX(layer['weights']),
                    border_mode=layer['border_mode'], **kwargs))
            elif layer['layer_type'] == 'Dense':
                self.snn.add(self.sim.SpikeDense(
                    layer['output_shape'][1],
                    self.sim.floatX(layer['weights']), **kwargs))
            elif layer['layer_type'] == 'Flatten':
                self.snn.add(self.sim.SpikeFlatten(**kwargs))
            elif layer['layer_type'] in {'MaxPooling2D', 'AveragePooling2D'}:
                self.snn.add(self.sim.AvgPool2DReLU(
                    pool_size=layer['pool_size'], strides=layer['strides'],
                    border_mode=layer['border_mode'], label=layer['label']))
            if layer['layer_type'] in {'Convolution2D', 'Dense', 'Flatten',
                                       'MaxPooling2D', 'AveragePooling2D'}:
                self.sim.init_neurons(self.snn.layers[-1],
                                      v_thresh=settings['v_thresh'],
                                      tau_refrac=settings['tau_refrac'],
                                      **kwargs2)

        # Compile
        echo('\n')
        echo("Compiling spiking network...\n")
        self.snn.compile(loss='categorical_crossentropy', optimizer='sgd',
                         metrics=['accuracy'])
        output_spikes = self.snn.layers[-1].get_output()
        output_time = self.sim.get_time(self.snn.layers[-1])
        updates = self.sim.get_updates(self.snn.layers[-1])
        self.get_output = theano.function([self.snn.input, input_time],
                                          [output_spikes, output_time],
                                          updates=updates)
        echo("Compilation finished.\n\n")

    def run(self, net, X_test, Y_test):
        """
        Simulate a spiking network with leaky integrate-and-fire units and
        Poisson input, using mean pooling and a timestepped approach.

        If ``globalparams['verbose'] > 1``, the simulator plots the spiketrains
        and spikerates of each neuron in each layer, for all
        ``globalparams['batch_size']`` samples of the first batch of
        ``X_test``.
        This is of course very costly in terms of memory and time, but can be
        useful for debugging the network's general functioning. To get detailed
        information about the network's behavior for a particular sample,
        replace the test set ``X_test, Y_test`` with this sample of interest.

        Parameters
        ----------

        X_test : float32 array
            The input samples to test.
            With data of the form (channels, num_rows, num_cols),
            X_test has dimension (num_samples, channels*num_rows*num_cols)
            for a multi-layer perceptron, and
            (num_samples, channels, num_rows, num_cols) for a convolutional
            net.
        Y_test : float32 array
            Ground truth of test data. Has dimension (num_samples, num_classes)
        snn : Keras model, possible kwarg
            The network architecture and weights in json format.
        get_output : Theano function, possible kwarg
            Gets the output of the network.

        Returns
        -------

        total_acc : float
            Number of correctly classified samples divided by total number of
            test samples.
        spiketrains : list of tuples
            Each entry in ``spiketrains`` contains a tuple
            ``(spiketimes, label)`` for each layer of the network (for the
            first batch only).
            ``spiketimes`` is a 2D array where the first index runs over the
            number of neurons in the layer, and the second index contains the
            spike times of the specific neuron.
            ``label`` is a string specifying both the layer type and the index,
            e.g. ``'03Dense'``.

        """

        import numpy as np

        # Load neuron layers and connections if conversion was done during a
        # previous session.
        if self.get_output is None:
            if settings['verbose'] > 1:
                echo("Restoring layer connections...\n")
            self.load('snn_keras_' + settings['filename'])

        # Ground truth
        truth = np.argmax(Y_test, axis=1)

        # This factor determines the probability threshold for cells in the
        # input layer to fire a spike. Increasing ``max_f`` increases the
        # firing rate.
        rescale_fac = 1000 / (settings['max_f'] * settings['dt'])

        # Divide the test set into batches and run all samples in a batch in
        # parallel.
        output = np.zeros(Y_test.shape).astype('int32')
        num_batches = int(np.ceil(X_test.shape[0] / settings['batch_size']))
        if settings['verbose'] > 1:
            echo("Starting new simulation...\n")
        # Allocate a list 'spiketrains' with the following specification:
        # Each entry in ``spiketrains`` contains a tuple
        # ``(spiketimes, label)`` for each layer of the network (for the first
        # batch only, and excluding ``Flatten`` layers).
        # ``spiketimes`` is an array where the first indices run over the
        # number of neurons in the layer, and the last index contains the spike
        # times of the specific neuron.
        # ``label`` is a string specifying both the layer type and the index,
        # e.g. ``'03Dense'``.
        spiketrains_batch = []
        for layer in self.snn.layers:
            if 'Flatten' not in layer.name:
                shape = list(layer.output_shape) + [int(settings['duration'] /
                                                        settings['dt'])]
                spiketrains_batch.append((np.zeros(shape), layer.name))

        for batch_idx in range(num_batches):
            # Determine batch indices.
            max_idx = min((batch_idx + 1) * settings['batch_size'],
                          Y_test.shape[0])
            batch_idxs = range(batch_idx * settings['batch_size'], max_idx)
            batch = X_test[batch_idxs, :]

            # Reset network variables.
            self.sim.reset(self.snn.layers[-1])

            # Loop through simulation time.
            t_idx = 0
            for t in np.arange(0, settings['duration'], settings['dt']):
                # Create poisson input.
                spike_snapshot = np.random.random_sample(batch.shape) * \
                    rescale_fac
                inp_images = (spike_snapshot <= batch).astype('float32')
                # Main step: Propagate poisson input through network and record
                # output spikes.
                out_spikes, ts = self.get_output(inp_images, float(t))
                # For the first batch only, record the spiketrains of each
                # neuron in each layer.
                if batch_idx == 0 and settings['verbose'] > 1:
                    j = 0
                    for i in range(len(self.snn.layers)):
                        if 'Flatten' not in self.snn.layers[i].name:
                            spiketrains_batch[j][0][Ellipsis, t_idx] = \
                                self.snn.layers[i].spiketrain.get_value()
                            j += 1
                t_idx += 1
                # Count number of spikes in output layer during whole
                # simulation.
                output[batch_idxs, :] += out_spikes.astype('int32')
                if settings['verbose'] > 1:
                    echo('.')
            if settings['verbose'] > 1:
                # Get result by comparing the guessed class (i.e. the index of
                # the neuron in the last layer which spiked most) to the ground
                # truth.
                guesses = np.argmax(output, axis=1)
                echo('\n')
                echo("Batch {} of {} completed ({:.1%})\n".format(
                    batch_idx + 1, num_batches, (batch_idx + 1) / num_batches))
                echo("Moving average accuracy: {:.2%}.\n".format(
                      np.sum(guesses[:max_idx] == truth[:max_idx]) / max_idx))
                if batch_idx == 0 and settings['verbose'] > 2:
                    output_graphs(spiketrains_batch, net, batch,
                                  settings['log_dir_of_current_run'])

        guesses = np.argmax(output, axis=1)
        total_acc = np.mean(guesses == truth)
        echo("Simulation finished.\n\n")
        echo("Total output spikes: {}\n".format(np.sum(output)))
        echo("Average output spikes per sample: {:.2f}\n".format(
             np.mean(np.sum(output, axis=1))))
        echo("Final time: {:.3f}\n".format(ts.max()))
        echo("Total accuracy: {:.2%} on {} test samples.\n\n".format(total_acc,
             output.shape[0]))

        return total_acc

    def save(self, path, filename):
        """
        Write model architecture and weights to disk.

        Parameters
        ----------

        model : network object
            The network model object in the ``model_lib`` language, e.g. keras.
        """

        from snntoolbox.io_utils.save import confirm_overwrite

        echo("Saving model to {}\n".format(path))

        filepath = os.path.join(path, filename + '.json')
        if confirm_overwrite(filepath):
            open(filepath, 'w').write(self.snn.to_json())
        self.snn.save_weights(
            os.path.join(path, os.path.join(path, filename + '.h5')),
            overwrite=settings['overwrite'])
        echo("Done.\n")

    def load(self, filename):
        from keras import models
        from snntoolbox.core.inisim import SpikeFlatten, SpikeDense
        from snntoolbox.core.inisim import SpikeConv2DReLU, AvgPool2DReLU
        custom_layers = {'SpikeFlatten': SpikeFlatten,
                         'SpikeDense': SpikeDense,
                         'SpikeConv2DReLU': SpikeConv2DReLU,
                         'AvgPool2DReLU': AvgPool2DReLU}

        path = os.path.join(settings['path'], filename + '.json')
        self.snn = models.model_from_json(open(path).read(),
                                          custom_objects=custom_layers)
        self.snn.load_weights(os.path.join(settings['path'], filename + '.h5'))

        # Allocate input variables
        input_time = theano.tensor.scalar('time')
        input_shape = list(self.snn.input_shape)
        input_shape[0] = settings['batch_size']
        self.snn.layers[0].batch_input_shape = input_shape
        kwargs = {'time_var': input_time}
        for layer in self.snn.layers:
            self.sim.init_neurons(layer, v_thresh=settings['v_thresh'],
                                  tau_refrac=settings['tau_refrac'], **kwargs)
            kwargs = {}

        # Compile model
        # Todo: Allow user to specify loss function here (optimizer is not
        # relevant as we do not train any more). Unfortunately, Keras does not
        # save these parameters. They can be obtained from the compiled model
        # by calling 'model.loss' and 'model.optimizer'.
        self.snn.compile(loss='categorical_crossentropy', optimizer='sgd',
                         metrics=['accuracy'])
        output_spikes = self.snn.layers[-1].get_output()
        output_time = self.sim.get_time(self.snn.layers[-1])
        updates = self.sim.get_updates(self.snn.layers[-1])
        self.get_output = theano.function([self.snn.input, input_time],
                                          [output_spikes, output_time],
                                          updates=updates)

    def end_sim(self):
        pass
