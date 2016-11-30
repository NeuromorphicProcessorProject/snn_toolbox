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
from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import os
import numpy as np
from textwrap import dedent
import keras
from future import standard_library
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

    snn: keras.models.Model
        Keras model. This is the output format of the compiled spiking model
        because INI simulator runs networks of layers that are derived from
        Keras layer base classes.

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
        self.snn = None
        self.parsed_model = None

    # noinspection PyUnusedLocal
    def build(self, parsed_model, **kwargs):
        """Compile a SNN to prepare for simulation with INI simulator.

        Convert an ANN to a spiking neural network, using layers derived from
        Keras base classes.

        Aims at simulating the network on a self-implemented Integrate-and-Fire
        simulator using a timestepped approach.

        Sets the ``snn`` attribute of this class.

        Parameters
        ----------

        parsed_model: Keras model
            Parsed input model; result of applying
            ``model_lib.extract(input_model)`` to the ``input model``.
        """

        self.parsed_model = parsed_model

        print("Building spiking model...")
        # Pass time variable to first layer
        input_images = keras.layers.Input(
            batch_shape=parsed_model.layers[0].batch_input_shape)
        spiking_layers = {parsed_model.layers[0].name: input_images}

        # Iterate over layers to create spiking neurons and connections.
        for layer in parsed_model.layers[1:]:  # Skip input layer
            print("Building layer: {}".format(layer.name))
            spike_layer = getattr(self.sim, 'Spike' + layer.__class__.__name__)
            inbound = [spiking_layers[inb.name] for inb in
                       layer.inbound_nodes[0].inbound_layers]
            spiking_layers[layer.name] = \
                spike_layer.from_config(layer.get_config())(inbound)

        print("Compiling spiking model...\n")
        self.snn = keras.models.Model(
            input_images, spiking_layers[parsed_model.layers[-1].name])
        self.snn.compile('sgd', 'categorical_crossentropy',
                         metrics=['accuracy'])
        self.snn.set_weights(parsed_model.get_weights())

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

        top1acc_total: float
            Number of correctly classified samples divided by total number of
            test samples.
        """

        from ann_architectures.imagenet.utils import preprocess_input
        from snntoolbox.core.util import get_activations_batch, get_top5score
        from snntoolbox.core.util import echo
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
        if self.snn is None:
            print("Restoring spiking network...\n")
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
        if s['verbose'] > 1:
            spiketrains_batch = []
            for layer in self.snn.layers:
                if not hasattr(layer, 'spiketrain'):
                    continue
                shape = list(layer.output_shape) + [int(s['duration']/s['dt'])]
                spiketrains_batch.append((np.zeros(shape, 'float32'),
                                          layer.name))

        inp = None
        top1err_vs_time = []
        top5score_moving = 0
        truth = []
        guesses = []
        guesses_batch = None
        rescale_fac = 1
        num_classes = 0
        activations_batch = None
        # Prepare files to write moving accuracy and error to.
        path_acc = os.path.join(log_dir, 'accuracy.txt')
        if os.path.isfile(path_acc):
            os.remove(path_acc)
        f_acc = open(path_acc, 'a')
        path_top5acc = os.path.join(log_dir, 'top5accuracy.txt')
        if os.path.isfile(path_top5acc):
            os.remove(path_top5acc)
        f_top5acc = open(path_top5acc, 'a')
        path_err = os.path.join(log_dir, 'err_vs_time.txt')
        if os.path.isfile(path_err):
            os.remove(path_err)
        f_err = open(path_err, 'a')

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
            self.reset()

            # Loop through simulation time.
            output = np.zeros((s['batch_size'], y_batch.shape[1]),
                              dtype='int32')
            num_timesteps = int(s['duration'] / s['dt'])
            if s['verbose'] > 1:
                total_spike_count_over_time = np.zeros((num_timesteps,
                                                        s['batch_size']))
            total_spike_count = np.zeros(s['batch_size'])
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
                self.set_time(t)
                out_spikes = self.snn.predict_on_batch(inp)
                output += out_spikes.astype('int32')
                # Get result by comparing the guessed class (i.e. the index
                # of the neuron in the last layer which spiked most) to the
                # ground truth.
                truth_batch = np.argmax(y_batch, axis=1)
                guesses_batch = np.argmax(output, axis=1)
                top1err_vs_time.append(np.mean(truth_batch != guesses_batch))
                # Record the spiketrains of each neuron in each layer.
                if s['verbose'] > 1:
                    j = 0
                    for layer in self.snn.layers:
                        if not hasattr(layer, 'spiketrain'):
                            continue
                        spiketrains_batch[j][0][Ellipsis, t_idx] = \
                            layer.spiketrain.get_value()
                        total_spike_count += layer.total_spike_count.get_value()
                        j += 1
                    total_spike_count_over_time[t_idx] = total_spike_count
                    t_idx += 1
                if s['verbose'] > 0:
                    echo('{:.2%}_'.format(1-top1err_vs_time[-1]))

            truth += list(truth_batch)
            guesses += list(guesses_batch)
            top1acc_moving = np.mean(np.array(truth) == np.array(guesses))
            top5score_moving += get_top5score(truth_batch, output)
            top5acc_moving = top5score_moving / ((batch_idx + 1) *
                                                 settings['batch_size'])
            with open(os.path.join(log_dir, 'predictions'), 'w') as f_pred:
                f_pred.write(str(guesses))
            with open(os.path.join(log_dir, 'target_classes'), 'w') as f_target:
                f_target.write(str(truth))
            f_acc.write("Moving average accuracy after batch {}/{}: {:.2%}."
                        "\n".format(batch_idx + 1, num_batches, top1acc_moving))
            f_top5acc.write("Moving average of top-5-accuracy after batch "
                            "{}/{}: {:.2%}.\n".format(
                                batch_idx + 1, num_batches, top5acc_moving))
            f_err.write(str(top1err_vs_time))

            if s['verbose'] > 0:
                print("\nBatch {} of {} completed ({:.1%})".format(
                    batch_idx + 1, num_batches, (batch_idx + 1) / num_batches))
                print("Moving average accuracy: {:.2%}.\n".format(
                    top1acc_moving))
                print("Moving top-5 accuracy: {:.2%}.\n".format(top5acc_moving))

            if s['verbose'] > 1:
                print("Saving batch activations...")
                activations_batch = get_activations_batch(self.parsed_model,
                                                          x_batch)
                path_activ = os.path.join(log_dir, 'activations')
                if not os.path.isdir(path_activ):
                    os.makedirs(path_activ)
                np.savez_compressed(os.path.join(path_activ, str(batch_idx)),
                                    activations=activations_batch)
                print("Saving batch spiketrains...")
                path_trains = os.path.join(log_dir, 'spiketrains')
                if not os.path.isdir(path_trains):
                    os.makedirs(path_trains)
                np.savez_compressed(os.path.join(path_trains, str(batch_idx)),
                                    spiketrains=spiketrains_batch)
                print("Saving batch spikecounts...")
                path_count = os.path.join(log_dir, 'spikecounts')
                if not os.path.isdir(path_count):
                    os.makedirs(path_count)
                np.savez_compressed(os.path.join(path_count, str(batch_idx)),
                                    spikecounts=total_spike_count_over_time)
            if s['verbose'] > 2:
                plot_input_image(x_batch[0], int(np.argmax(y_batch[0])),
                                 log_dir)
                ann_err = self.ANN_err if hasattr(self, 'ANN_err') else None
                plot_error_vs_time(top1err_vs_time, ann_err=ann_err,
                                   path=log_dir)
                plot_spikecount_vs_time(total_spike_count_over_time, log_dir)
                plot_confusion_matrix(truth, guesses, log_dir)
                output_graphs(spiketrains_batch, activations_batch, log_dir)
        f_acc.close()
        f_top5acc.close()
        f_err.close()

        count = np.zeros(num_classes)
        match = np.zeros(num_classes)
        for gt, p in zip(truth, guesses):
            count[gt] += 1
            if gt == p:
                match[gt] += 1
        avg_acc = np.mean(match / count)
        top1acc_total = np.mean(np.array(truth) == np.array(guesses))
        if s['verbose'] > 2:
            plot_confusion_matrix(truth, guesses, log_dir)
        print("Simulation finished.\n\n")
        print("Total accuracy: {:.2%} on {} test samples.\n\n".format(
            top1acc_total, len(guesses)))
        print("Accuracy averaged over classes: {}".format(avg_acc))

        return top1acc_total

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

        print("Saving model to {}...\n".format(filepath))
        self.snn.save(filepath, settings['overwrite'])

    def load(self, path=None, filename=None):
        """Load model architecture and parameters from disk.

        Sets the ``snn`` attribute of this class.

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

    def set_time(self, t):
        """Set the simulation time variable of all layers in the network.

        Parameters
        ----------

        t: float
            Current simulation time.
        """

        for layer in self.snn.layers[1:]:
            if self.sim.get_time(layer) is not None:  # Has time attribute
                self.sim.set_time(layer, np.float32(t))

    def reset(self):
        """Reset network variables."""

        for layer in self.snn.layers[1:]:  # Skip input layer
            layer.reset()
