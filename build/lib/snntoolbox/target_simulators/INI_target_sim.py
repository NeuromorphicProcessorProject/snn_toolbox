# -*- coding: utf-8 -*-
"""Building SNNs using INI simulator.

The modules in ``target_simulators`` package allow building a spiking network
and exporting it for use in a spiking simulator.

This particular module offers functionality for the INI simulator. Adding
another simulator requires implementing the class ``SNN_compiled`` with its
methods tailored to the specific simulator.

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
import keras
from textwrap import dedent
from snntoolbox import echo
from snntoolbox.config import settings, initialize_simulator

standard_library.install_aliases()

if settings['online_normalization']:
    lidx = 0


class SNN():
    """
    The compiled spiking neural network, ready for testing in a spiking
    simulator.

    Attributes
    ----------

    sim: Simulator
        Module containing utility functions of spiking simulator. Result of
        calling ``snntoolbox.config.initialize_simulator()``. For instance, if
        using Brian simulator, this initialization would be equivalent to
        ``import pyNN.brian as sim``.

    snn: Spiking Model
        Keras ``Sequential`` model. This is the output format of the compiled
        spiking model because INI simulator runs networks of layers that are
        derived from Keras layer base classes.

    get_output: Theano function
        Compute output of network by iterating over all layers.

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
        Clean up after simulation. Not needed in this simulator, so do a
        ``pass``.
    """

    def __init__(self):
        """Init function."""
        self.sim = initialize_simulator()
        self.snn = keras.models.Sequential()
        self.get_output = None

    def build(self, parsed_model):
        """Compile a SNN to prepare for simulation with INI simulator.

        Convert an ANN to a spiking neural network, using layers derived from
        Keras base classes.

        Aims at simulating the network on a self-implemented Integrate-and-Fire
        simulator using a timestepped approach.

        Sets the ``snn`` and ``get_output`` attributes of this class.

        Parameters
        ----------

        parsed_model: Keras model
            Parsed input model; result of applying
            ``model_lib.extract(input_model)`` to the ``input model``.
        """

        self.parsed_model = parsed_model

        echo('\n' + "Compiling spiking network...\n")

        # Pass time variable to first layer
        input_time = theano.tensor.scalar('time')
        kwargs = {'time_var': input_time}

        # Iterate over layers to create spiking neurons and connections.
        for layer in parsed_model.layers:
            echo("Building layer: {}\n".format(layer.name))
            spike_layer = getattr(self.sim, 'Spike' + layer.__class__.__name__)
            self.snn.add(spike_layer(**layer.get_config()))
            self.snn.layers[-1].set_weights(layer.get_weights())
            self.sim.init_neurons(self.snn.layers[-1],
                                  v_thresh=settings['v_thresh'],
                                  tau_refrac=settings['tau_refrac'],
                                  **kwargs)
            kwargs = {}

        # Compile
        self.compile_snn(input_time)

    def compile_snn(self, input_time):
        """Set the ``snn`` and ``get_output`` attributes of this class."""

        # Optimizer and loss are required by the compiler but not needed at
        # inference time, so we simply set it to the most common choice here.
        self.snn.compile('sgd', 'categorical_crossentropy',
                         metrics=['accuracy'])
        output_spikes = self.snn.layers[-1].get_output()
        output_time = self.sim.get_time(self.snn.layers[-1])
        updates = self.sim.get_updates(self.snn.layers[-1])
        if settings['online_normalization']:
            thresh = self.snn.layers[lidx].v_thresh
            max_spikerate = self.snn.layers[lidx].max_spikerate
            spiketrain = self.snn.layers[lidx].spiketrain
            self.get_output = theano.function([self.snn.input, input_time],
                                              [output_spikes, output_time,
                                               thresh, max_spikerate,
                                               spiketrain], updates=updates,
                                              allow_input_downcast=True)
        else:
            self.get_output = theano.function([self.snn.input, input_time],
                                              [output_spikes, output_time],
                                              updates=updates,
                                              allow_input_downcast=True)

    def run(self, X_test, Y_test):
        """Simulate a SNN with LIF and Poisson input.

        Simulate a spiking network with leaky integrate-and-fire units and
        Poisson input, using mean pooling and a timestepped approach.

        If ``settings['verbose'] > 1``, the toolbox plots the spiketrains
        and spikerates of each neuron in each layer, for the first sample of
        the first batch of ``X_test``.

        This is somewhat costly in terms of memory and time, but can be useful
        for debugging the network's general functioning.

        Parameters
        ----------

        X_test: float32 array
            The input samples to test.
            With data of the form (channels, num_rows, num_cols),
            X_test has dimension (num_samples, channels*num_rows*num_cols)
            for a multi-layer perceptron, and
            (num_samples, channels, num_rows, num_cols) for a convolutional
            net.
        Y_test: float32 array
            Ground truth of test data. Has dimension (num_samples, num_classes)

        Returns
        -------

        total_acc: float
            Number of correctly classified samples divided by total number of
            test samples.
        """

        import numpy as np
        from snntoolbox.core.util import get_activations_batch
        from snntoolbox.io_utils.plotting import output_graphs
        from snntoolbox.io_utils.plotting import plot_confusion_matrix
        from snntoolbox.io_utils.plotting import plot_error_vs_time

        # Load neuron layers and connections if conversion was done during a
        # previous session.
        if self.get_output is None:
            echo("Restoring layer connections...\n")
            self.load()
            self.parsed_model = keras.models.load_model(os.path.join(
                settings['path'], settings['filename_parsed_model'] + '.h5'))

        si = settings['sample_indices_to_test']
        if not si == []:
            assert len(si) == settings['batch_size'], dedent("""
                You attempted to test the SNN on a total number of samples that
                is not compatible with the batch size with which the SNN was
                converted. Either change the number of samples to test to be
                equal to the batch size, or convert the ANN again using the
                corresponding batch size.""")
            X_test = np.array([X_test[i] for i in si])
            Y_test = np.array([Y_test[i] for i in si])

        # Ground truth
        truth = np.argmax(Y_test, axis=1)

        # Divide the test set into batches and run all samples in a batch in
        # parallel.
        output = np.zeros(Y_test.shape).astype('int32')
        num_batches = int(np.ceil(X_test.shape[0] / settings['batch_size']))
        if settings['verbose'] > 1:
            print("Starting new simulation...\n")
        # Allocate a list 'spiketrains_batch' with the following specification:
        # Each entry in ``spiketrains_batch`` contains a tuple
        # ``(spiketimes, label)`` for each layer of the network (for the first
        # batch only, and excluding ``Flatten`` layers).
        # ``spiketimes`` is an array where the last index contains the spike
        # times of the specific neuron, and the first indices run over the
        # number of neurons in the layer:
        # (batch_size, n_chnls, n_rows, n_cols, duration)
        # ``label`` is a string specifying both the layer type and the index,
        # e.g. ``'03Dense'``.
        if settings['verbose'] > 2:
            spiketrains_batch = []
            for layer in self.snn.layers:
                if 'Flatten' in layer.name:
                    continue
                shape = list(layer.output_shape) + \
                    [int(settings['duration'] / settings['dt'])]
                spiketrains_batch.append((np.zeros(shape, 'float32'),
                                          layer.name))
            # Allocate list for plotting the error vs simulation time
            err = []

        count = np.zeros(Y_test.shape[1])
        match = np.zeros(Y_test.shape[1])
        for batch_idx in range(num_batches):  # m=2.3GB
            # Determine batch indices.
            max_idx = min((batch_idx + 1) * settings['batch_size'],
                          Y_test.shape[0])
            batch_idxs = range(batch_idx * settings['batch_size'], max_idx)
            batch = X_test[batch_idxs, :]

            # Either use Poisson spiketrains as inputs to the SNN, or take the
            # original data.
            if settings['poisson_input']:
                # This factor determines the probability threshold for cells in
                # the input layer to fire a spike. Increasing ``input_rate``
                # increases the firing rate of the input and subsequent layers.
                rescale_fac = 1000 / (settings['input_rate'] * settings['dt'])
            else:
                # Simply use the analog values of the original data as input.
                inp = batch

            # Reset network variables.
            self.sim.reset(self.snn.layers[-1])

            # Loop through simulation time.
            t_idx = 0
            for t in np.arange(0, settings['duration'], settings['dt']):
                if settings['poisson_input']:
                    # Create poisson input.
                    spike_snapshot = np.random.random_sample(batch.shape) * \
                        rescale_fac
                    inp = (spike_snapshot <= batch).astype('float32')
                # Main step: Propagate poisson input through network and record
                # output spikes.
                if settings['online_normalization']:
                    out_spikes, ts, thresh, max_spikerate, spiketrain = \
                        self.get_output(inp, float(t))
                    print('Time: {:.2f}, thresh: {:.2f},'
                          ' max_spikerate: {:.2f}'.format(
                            float(np.array(ts)),
                            float(np.array(thresh)),
                            float(np.array(max_spikerate))))
                else:
                    out_spikes, ts = self.get_output(inp, float(t))  # t=27.6%
                # For the first batch only, record the spiketrains of each
                # neuron in each layer.
                if batch_idx == 0 and settings['verbose'] > 2:
                    j = 0
                    for i, layer in enumerate(self.snn.layers):
                        if 'Flatten' not in self.snn.layers[i].name:
                            spiketrains_batch[j][0][Ellipsis, t_idx] = \
                                layer.spiketrain.get_value()  # t=1.8% m=0.6GB
                            j += 1
                t_idx += 1
                # Count number of spikes in output layer during whole
                # simulation.
                output[batch_idxs, :] += out_spikes.astype('int32')
                if batch_idx == 0 and settings['verbose'] > 2:
                    # Get result by comparing the guessed class (i.e. the index
                    # of the neuron in the last layer which spiked most) to the
                    # ground truth.
                    guesses = np.argmax(output, axis=1)
                    err.append(
                        np.sum(truth[:max_idx] != guesses[:max_idx]) / max_idx)
                if settings['verbose'] > 1:
                    echo('.')

            if settings['verbose'] > 0:
                echo('\n')
                echo("Batch {} of {} completed ({:.1%})\n".format(
                    batch_idx + 1, num_batches, (batch_idx + 1) / num_batches))
                guesses = np.argmax(output, axis=1)
                avg = np.sum(truth[:max_idx] == guesses[:max_idx]) / max_idx
                echo("Moving average accuracy: {:.2%}.\n".format(avg))
                with open(os.path.join(settings['log_dir_of_current_run'],
                                       'accuracy.txt'), 'w') as f:
                    f.write("Moving average accuracy after batch " +
                            "{} of {}: {:.2%}.\n".format(batch_idx + 1,
                                                         num_batches, avg))
                if batch_idx == 0 and settings['verbose'] > 2:
                    plot_error_vs_time(err, settings['log_dir_of_current_run'])
                    plot_confusion_matrix(truth[:max_idx], guesses[:max_idx],
                                          settings['log_dir_of_current_run'])
                    activations_batch = get_activations_batch(
                        self.parsed_model, batch)
                    output_graphs(spiketrains_batch, activations_batch,
                                  settings['log_dir_of_current_run'])
                    # t=70.1% m=0.6GB
                    del spiketrains_batch

        guesses = np.argmax(output, axis=1)
        for gt, p in zip(truth, guesses):
            count[gt] += 1
            if gt == p:
                match[gt] += 1
        avg_acc = np.mean(match / count)
        total_acc = np.mean(truth == guesses)
        if settings['verbose'] > 2:
            plot_confusion_matrix(truth, guesses,
                                  settings['log_dir_of_current_run'])
        echo("Simulation finished.\n\n")
        echo("Total output spikes: {}\n".format(np.sum(output)))
        echo("Average output spikes per sample: {:.2f}\n".format(
             np.mean(np.sum(output, axis=1))))
        echo("Total accuracy: {:.2%} on {} test samples.\n\n".format(total_acc,
             output.shape[0]))
        echo("Accuracy averaged over classes: {}".format(avg_acc))

        return total_acc

    def save(self, path=None, filename=None):
        """Write model architecture and parameters to disk.

        Parameters
        ----------

        path: string, optional
            Path to directory where to save model to. Defaults to
            ``settings['path']``.

        filename: string, optional
            Name of file to write model to. Defaults to
            ``settings['filename_snn']``.
        """

        if path is None:
            path = settings['path']
        if filename is None:
            filename = settings['filename_snn']
        filepath = os.path.join(path, filename + '.h5')

        echo("Saving model to {}...\n".format(filepath))
        self.snn.save(filepath, settings['overwrite'])

    def load(self, path=None, filename=None):
        """Load model architecture and parameters from disk.

        Sets the ``snn`` and ``get_output`` attributes of this class.

        Parameters
        ----------

        path: string, optional
            Path to directory where to load model from. Defaults to
            ``settings['path']``.

        filename: string, optional
            Name of file to load model from. Defaults to
            ``settings['filename_snn']``.
        """

        from snntoolbox.core.inisim import custom_layers

        if path is None:
            path = settings['path']
        if filename is None:
            filename = settings['filename_snn']
        filepath = os.path.join(path, filename + '.h5')

        self.snn = keras.models.load_model(filepath, custom_layers)
        self.assert_batch_size(self.snn.layers[0].batch_input_shape[0])

        # Allocate input variables
        input_time = theano.tensor.scalar('time')
        kwargs = {'time_var': input_time}
        for layer in self.snn.layers:
            self.sim.init_neurons(layer, v_thresh=settings['v_thresh'],
                                  tau_refrac=settings['tau_refrac'], **kwargs)
            kwargs = {}

        # Compile model
        self.compile_snn(input_time)

    def assert_batch_size(self, batch_size):
        """Check if batchsize is matched with configuration."""

        if batch_size != settings['batch_size']:
            msg = dedent("""\
                You attempted to use the SNN with a batch_size different than
                the one with which it was converted. This is not supported when
                using INI simulator: To change the batch size, convert the ANN
                from scratch with the desired batch size. For now, the batch
                size has been reset from {} to the original {}.\n""".format(
                settings['batch_size'], batch_size))
            # logging.warning(msg)
            print(msg)
            settings['batch_size'] = batch_size

    def end_sim(self):
        """Clean up after simulation.

        Clean up after simulation. Not needed in this simulator, so do a
        ``pass``.
        """

        pass
