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

import os
import theano
import keras
from textwrap import dedent
from snntoolbox import echo
from snntoolbox.config import settings, initialize_simulator

standard_library.install_aliases()

if settings['online_normalization']:
    lidx = 0


class SNN:
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

    def __init__(self, s=None):
        """Init function."""

        if s is None:
            s = settings
        self.sim = initialize_simulator(s['simulator'])
        self.snn = keras.models.Sequential()
        self.get_output = None
        self.parsed_model = None

    # noinspection PyUnusedLocal
    def build(self, parsed_model, **kwargs):
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
        kwargs2 = {'time_var': input_time}

        # Iterate over layers to create spiking neurons and connections.
        for layer in parsed_model.layers:
            echo("Building layer: {}\n".format(layer.name))
            spike_layer = getattr(self.sim, 'Spike' + layer.__class__.__name__)
            self.snn.add(spike_layer(**layer.get_config()))
            self.snn.layers[-1].set_weights(layer.get_weights())
            self.sim.init_neurons(self.snn.layers[-1],
                                  v_thresh=settings['v_thresh'],
                                  tau_refrac=settings['tau_refrac'],
                                  **kwargs2)
            kwargs2 = {}

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
        total_spike_count = self.snn.layers[0].total_spike_count
        for layer in self.snn.layers[1:]:
            total_spike_count += layer.total_spike_count
        updates = self.sim.get_updates(self.snn.layers[-1])
        if settings['online_normalization']:
            thresh = self.snn.layers[lidx].v_thresh
            max_spikerate = self.snn.layers[lidx].max_spikerate
            spiketrain = self.snn.layers[lidx].spiketrain
            self.get_output = theano.function([self.snn.input, input_time],
                                              [output_spikes, output_time,
                                               total_spike_count,
                                               thresh, max_spikerate,
                                               spiketrain], updates=updates,
                                              allow_input_downcast=True)
        else:
            self.get_output = theano.function([self.snn.input, input_time],
                                              [output_spikes, output_time,
                                               total_spike_count],
                                              updates=updates,
                                              allow_input_downcast=True)

    def run(self, x_test=None, y_test=None, dataflow=None, **kwargs):
        """Simulate a SNN with LIF and Poisson input.

        Simulate a spiking network with leaky integrate-and-fire units and
        Poisson input, using mean pooling and a timestepped approach.

        If ``settings['verbose'] > 1``, the toolbox plots the spiketrains
        and spikerates of each neuron in each layer, for the first sample of
        the first batch of ``x_test``.

        This is somewhat costly in terms of memory and time, but can be useful
        for debugging the network's general functioning.

        Parameters
        ----------

        x_test: float32 array
            The input samples to test.
            With data of the form (channels, num_rows, num_cols),
            x_test has dimension (num_samples, channels*num_rows*num_cols)
            for a multi-layer perceptron, and
            (num_samples, channels, num_rows, num_cols) for a convolutional
            net.
        y_test: float32 array
            Ground truth of test data. Has dimension (num_samples, num_classes)
        dataflow : keras.DataFlowGenerator

        kwargs: Optional[dict]
            - s: Optional[dict]
                Settings. If not given, the ``snntoolobx.config.settings``
                dictionary is used.
            - path: Optional[str]
                Where to store the output plots. If no path given, this value is
                taken from the settings dictionary.

        Returns
        -------

        total_acc: float
            Number of correctly classified samples divided by total number of
            test samples.
        """

        import numpy as np
        from ann_architectures.imagenet.utils import preprocess_input
        from snntoolbox.core.util import get_activations_batch
        from snntoolbox.io_utils.plotting import output_graphs
        from snntoolbox.io_utils.plotting import plot_confusion_matrix
        from snntoolbox.io_utils.plotting import plot_error_vs_time
        from snntoolbox.io_utils.plotting import plot_input_image
        from snntoolbox.io_utils.plotting import plot_spikecount_vs_time

        s = kwargs['settings'] if 'settings' in kwargs else settings
        log_dir = kwargs['path'] if 'path' in kwargs \
            else s['log_dir_of_current_run']

        # Load neuron layers and connections if conversion was done during a
        # previous session.
        if self.get_output is None:
            echo("Restoring layer connections...\n")
            self.load()
            self.parsed_model = keras.models.load_model(os.path.join(
                s['path_wd'], s['filename_parsed_model']+'.h5'))

        si = s['sample_indices_to_test'] \
            if 'sample_indices_to_test' in s else []
        if not si == []:
            assert len(si) == s['batch_size'], dedent("""
                You attempted to test the SNN on a total number of samples that
                is not compatible with the batch size with which the SNN was
                converted. Either change the number of samples to test to be
                equal to the batch size, or convert the ANN again using the
                corresponding batch size.""")
            if x_test is None:
                # Probably need to turn off shuffling in ImageDataGenerator
                # for this to produce the desired samples.
                x_test, y_test = dataflow.next()
            x_test = np.array([x_test[i] for i in si])
            y_test = np.array([y_test[i] for i in si])

        # Divide the test set into batches and run all samples in a batch in
        # parallel.
        num_batches = int(np.floor(s['num_to_test'] / s['batch_size']))

        if s['verbose'] > 1:
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
        if s['verbose'] > 2:
            spiketrains_batch = []
            for layer in self.snn.layers:
                if 'Flatten' in layer.name:
                    continue
                shape = list(layer.output_shape) + [int(s['duration']/s['dt'])]
                spiketrains_batch.append((np.zeros(shape, 'float32'),
                                          layer.name))
            # Allocate list for plotting the error vs simulation time
            err = []

        record_error_vs_time = False
        if record_error_vs_time:
            err = []

        truth = []
        guesses = []
        rescale_fac = 1
        num_classes = 0
        for batch_idx in range(num_batches):
            # Get a batch of samples
            if x_test is None:
                x_batch, y_batch = dataflow.next()
                imagenet = False
                if imagenet:  # Only for imagenet!
                    x_batch = preprocess_input(x_batch)
            else:
                batch_idxs = range(s['batch_size'] * batch_idx,
                                   s['batch_size'] * (batch_idx + 1))
                x_batch = x_test[batch_idxs, :]
                y_batch = y_test[batch_idxs, :]
            num_classes = y_batch.shape[1]

            # Either use Poisson spiketrains as inputs to the SNN, or take the
            # original data.
            if s['poisson_input']:
                # This factor determines the probability threshold for cells in
                # the input layer to fire a spike. Increasing ``input_rate``
                # increases the firing rate of the input and subsequent layers.
                rescale_fac = np.max(x_batch) * 1000 / (s['input_rate']*s['dt'])
            else:
                # Simply use the analog values of the original data as input.
                inp = x_batch

            # Reset network variables.
            self.sim.reset(self.snn.layers[-1])

            # Loop through simulation time.
            output = np.zeros((s['batch_size'], y_batch.shape[1]),
                              dtype='int32')
            num_timesteps = int(s['duration'] / s['dt'])
            if s['verbose'] > 2:
                total_spike_count_over_time = np.zeros((num_timesteps,
                                                        s['batch_size']))

            t_idx = 0
            for t in np.arange(0, s['duration'], s['dt']):
                if s['poisson_input']:
                    # Create poisson input.
                    spike_snapshot = \
                        np.random.random_sample(x_batch.shape) * rescale_fac
                    inp = (spike_snapshot <= np.abs(x_batch)).astype('float32')
                    # For BinaryNets, with input that is not normalized and not
                    # all positive, we stimulate with spikes of the same size
                    # as the maximum activation, and the same sign as the
                    # corresponding activation. Is there a better solution?
                    # inp *= np.max(x_batch) * np.sign(x_batch)
                # Main step: Propagate poisson input through network and record
                # output spikes.
                if s['online_normalization']:
                    out_spikes, ts, total_spike_count, thresh, max_spikerate, \
                        spiketrain = self.get_output(inp, float(t))
                    print('Time: {:.2f}, thresh: {:.2f},'
                          ' max_spikerate: {:.2f}'.format(
                            float(np.array(ts)),
                            float(np.array(thresh)),
                            float(np.array(max_spikerate))))
                else:
                    out_spikes, ts, total_spike_count = \
                        self.get_output(inp, float(t))
                # Count number of spikes in output layer during whole
                # simulation.
                output += out_spikes.astype('int32')
                # For the first batch only, record the spiketrains of each
                # neuron in each layer.
                verb = 1 if record_error_vs_time else 2
                if batch_idx == 0 and s['verbose'] > verb:
                    j = 0
                    for i, layer in enumerate(self.snn.layers):
                        if 'Flatten' in self.snn.layers[i].name:
                            continue
                        spiketrains_batch[j][0][Ellipsis, t_idx] = \
                            layer.spiketrain.get_value()
                        j += 1
                    total_spike_count_over_time[t_idx] = \
                        np.array(total_spike_count)
                    # Get result by comparing the guessed class (i.e. the index
                    # of the neuron in the last layer which spiked most) to the
                    # ground truth.
                    truth_tmp = np.argmax(y_batch, axis=1)
                    guesses_tmp = np.argmax(output, axis=1)
                    err.append(np.mean(truth_tmp != guesses_tmp))
                t_idx += 1
                if s['verbose'] > 1:
                    echo('.')

            if s['verbose'] > 0:
                echo('\n')
                echo("Batch {} of {} completed ({:.1%})\n".format(
                    batch_idx + 1, num_batches, (batch_idx + 1) / num_batches))
                truth += list(np.argmax(y_batch, axis=1))
                guesses += list(np.argmax(output, axis=1))
                avg = np.mean(np.array(truth) == np.array(guesses))
                echo("Moving average accuracy: {:.2%}.\n".format(avg))
                with open(os.path.join(log_dir, 'accuracy.txt'), 'w') as f:
                    f.write("Moving average accuracy after batch " +
                            "{} of {}: {:.2%}.\n".format(batch_idx + 1,
                                                         num_batches, avg))

                if record_error_vs_time:
                    ann_err = self.ANN_err if hasattr(self, 'ANN_err') else None
                    plot_error_vs_time(err, ann_err=ann_err, path=log_dir)
                    with open(os.path.join(log_dir, 'err_vs_time.txt'), 'w') \
                            as f:
                        f.write(str(err))

                if batch_idx == 0 and s['verbose'] > 2:
                    plot_input_image(x_batch[0], int(np.argmax(y_batch[0])),
                                     log_dir)
                    ann_err = self.ANN_err if hasattr(self, 'ANN_err') else None
                    plot_error_vs_time(err, ann_err=ann_err, path=log_dir)
                    with open(os.path.join(log_dir, 'err_vs_time.txt'), 'w') \
                            as f:
                        f.write(str(err))
                    plot_spikecount_vs_time(total_spike_count_over_time,
                                            log_dir)
                    plot_confusion_matrix(truth, guesses, log_dir)
                    activations_batch = get_activations_batch(
                        self.parsed_model, x_batch)
                    output_graphs(spiketrains_batch, activations_batch, log_dir)
                    # del spiketrains_batch

                save_activations = True
                if save_activations:
                    print("Saving activations")
                    activations_batch = get_activations_batch(
                        self.parsed_model, x_batch)
                    path = os.path.join(log_dir, 'activations')
                    if not os.path.isdir(path):
                        os.makedirs(path)
                    np.savez_compressed(os.path.join(path, str(batch_idx)),
                                        activations=activations_batch,
                                        spiketrains=spiketrains_batch)

        count = np.zeros(num_classes)
        match = np.zeros(num_classes)
        for gt, p in zip(truth, guesses):
            count[gt] += 1
            if gt == p:
                match[gt] += 1
        avg_acc = np.mean(match / count)
        total_acc = np.mean(np.array(truth) == np.array(guesses))
        if s['verbose'] > 2:
            plot_confusion_matrix(truth, guesses, log_dir)
        echo("Simulation finished.\n\n")
        echo("Total accuracy: {:.2%} on {} test samples.\n\n".format(total_acc,
             len(guesses)))
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
            path = settings['path_wd']
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

    @staticmethod
    def assert_batch_size(batch_size):
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

    @staticmethod
    def end_sim():
        """Clean up after simulation.

        Clean up after simulation. Not needed in this simulator, so do a
        ``pass``.
        """

        pass
