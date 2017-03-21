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
from snntoolbox.core.inisim import bias_relaxation
from typing import Optional

standard_library.install_aliases()

if settings['online_normalization']:
    lidx = 0

remove_classifier = False
use_dvs_input = False
nb_events_per_sample = 2000
labeldict = {'paper': '0', 'scissors': '1', 'rock': '2', 'background': '3'}


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
        self.debug_vars = None

    # noinspection PyUnusedLocal
    def build(self, parsed_model, verbose=True, **kwargs):
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
        verbose: Optional[bool]
            Whether or not to print status messages.
        """

        if verbose:
            print("Building spiking model...")

        self.parsed_model = parsed_model

        if 'batch_size' in kwargs:
            batch_shape = [kwargs['batch_size']] + \
                          list(parsed_model.layers[0].batch_input_shape)[1:]
        else:
            batch_shape = list(parsed_model.layers[0].batch_input_shape)
        if batch_shape[0] is None:
            batch_shape[0] = settings['batch_size']

        input_images = keras.layers.Input(batch_shape=batch_shape)
        spiking_layers = {parsed_model.layers[0].name: input_images}

        # Iterate over layers to create spiking neurons and connections.
        for layer in parsed_model.layers[1:]:  # Skip input layer
            if verbose:
                print("Building layer: {}".format(layer.name))
            spike_layer = getattr(self.sim, 'Spike' + layer.__class__.__name__)
            inbound = [spiking_layers[inb.name] for inb in
                       layer.inbound_nodes[0].inbound_layers]
            spiking_layers[layer.name] = \
                spike_layer.from_config(layer.get_config())(inbound)

        if verbose:
            print("Compiling spiking model...\n")
        self.snn = keras.models.Model(
            input_images, spiking_layers[parsed_model.layers[-1].name])
        self.snn.compile('sgd', 'categorical_crossentropy',
                         metrics=['accuracy'])
        self.snn.set_weights(parsed_model.get_weights())
        for layer in self.snn.layers:
            if hasattr(layer, 'b'):
                # Adjust biases to time resolution of simulator.
                layer.b.set_value(layer.b.get_value() * settings['dt'])
                if bias_relaxation:  # Experimental
                    layer.b0.set_value(layer.b.get_value())

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

        # from ann_architectures.imagenet.utils import preprocess_input
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
        num_batches = int(1e9) if use_dvs_input else \
            int(np.floor(s['num_to_test'] / s['batch_size']))
        num_timesteps = int(s['duration'] / s['dt'])
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
        if 'spiketrains' in s['log_vars']:
            spiketrains_batch = []
            for layer in self.snn.layers:
                if not hasattr(layer, 'spiketrain'):
                    continue
                shape = list(layer.output_shape) + [num_timesteps]
                spiketrains_batch.append((np.zeros(shape, 'float32'),
                                          layer.name))

        num_layers_with_spikes = len([1 for l in self.snn.layers
                                      if hasattr(l, 'spiketrain')])
        inp = None
        input_t = None
        top5score_moving = 0
        truth = []
        guesses = []
        guesses_batch = None
        rescale_fac = 1
        activations_batch = None
        x_batch_xaddr = x_batch_yaddr = x_batch_ts = None
        x_batch = y_batch = None
        dvs_gen = None
        num_classes = None
        if use_dvs_input:
            dvs_gen = DVSIterator(os.path.join(s['dataset_path'], 'DVS'),
                                  s['batch_size'], labeldict,
                                  (239 / 63, 179 / 63), nb_events_per_sample)

        # Prepare files to write moving accuracy and error to.
        path_pred = os.path.join(log_dir, 'predictions')
        if os.path.isfile(path_pred):
            os.remove(path_pred)
        path_target = os.path.join(log_dir, 'target_classes')
        if os.path.isfile(path_target):
            os.remove(path_target)
        path_acc = os.path.join(log_dir, 'accuracy.txt')
        if os.path.isfile(path_acc):
            os.remove(path_acc)
        path_top5acc = os.path.join(log_dir, 'top5accuracy.txt')
        if os.path.isfile(path_top5acc):
            os.remove(path_top5acc)
        path_activ = os.path.join(log_dir, 'activations')
        if not os.path.isdir(path_activ):
            os.makedirs(path_activ)
        path_trains = os.path.join(log_dir, 'spiketrains')
        if not os.path.isdir(path_trains):
            os.makedirs(path_trains)
        path_count = os.path.join(log_dir, 'spikecounts')
        if not os.path.isdir(path_count):
            os.makedirs(path_count)
        path_input = os.path.join(log_dir, 'input_t')
        if not os.path.isdir(path_input):
            os.makedirs(path_input)
        net_top1err_t = []

        for batch_idx in range(num_batches):
            # Get a batch of samples
            if x_test is None:
                x_batch, y_batch = dataflow.next()
                imagenet = True
                if imagenet:  # Only for imagenet!
                    print("Preprocessing input for ImageNet")
                    x_batch = np.add(np.multiply(x_batch, 2. / 255.), - 1.)
                    # x_batch = preprocess_input(x_batch)
            elif not use_dvs_input:
                batch_idxs = range(s['batch_size'] * batch_idx,
                                   s['batch_size'] * (batch_idx + 1))
                x_batch = x_test[batch_idxs, :]
                y_batch = y_test[batch_idxs, :]
            if use_dvs_input:
                try:
                    x_batch_xaddr, x_batch_yaddr, x_batch_ts, y_batch = \
                        dvs_gen.__next__()
                except StopIteration:
                    break
            truth_batch = np.argmax(y_batch, axis=1)
            num_classes = y_batch.shape[1]

            # Either use Poisson spiketrains as inputs to the SNN, or take the
            # original data.
            if s['poisson_input']:
                # This factor determines the probability threshold for cells in
                # the input layer to fire a spike. Increasing ``input_rate``
                # increases the firing rate of the input and subsequent layers.
                rescale_fac = np.max(x_batch) * 1000 / s['input_rate'] / s['dt']
            elif use_dvs_input:
                pass
            else:
                # Simply use the analog values of the original data as input.
                inp = x_batch * s['dt']
                # inp = np.random.random_sample(x_batch.shape)

            # Reset network variables.
            self.reset()

            # Allocate variables to monitor during simulation
            output = np.zeros((s['batch_size'], num_classes), 'int32')
            if 'spikecounts' in s['plot_vars'] + s['log_vars']:
                total_spikecount_t = np.zeros((num_timesteps, s['batch_size']))
            self.init_debug_vars()
            top1err_vs_time = []
            if 'input_t' in s['log_vars']:
                input_t = np.empty([num_timesteps] +
                                   list(self.snn.input_shape), 'int32')
            total_spikecount = None
            input_spikecount = 0
            layer_spikecounts = np.zeros(
                (s['batch_size'], num_layers_with_spikes, num_timesteps))
            sim_step_int = 0
            print("Starting new simulation...\n")
            # Loop through simulation time.
            for sim_step in range(0, s['duration'], s['dt']):
                # Generate input, in case it changes with each simulation step:
                if s['poisson_input']:
                    if True:  # input_spikecount < nb_events_per_sample:
                        spike_snapshot = \
                            np.random.random_sample(x_batch.shape) * rescale_fac
                        inp = (spike_snapshot <= np.abs(x_batch)).astype(
                            'float32')
                        input_spikecount += np.count_nonzero(inp) / len(x_batch)
                        # For BinaryNets, with input that is not normalized and
                        # not all positive, we stimulate with spikes of the same
                        # size as the maximum activation, and the same sign as
                        # the corresponding activation. Is there a better
                        # solution?
                        # inp *= np.max(x_batch) * np.sign(x_batch)
                    else:
                        inp = np.zeros(x_batch.shape)
                elif use_dvs_input:
                    # print("Generating a batch of even-frames...")
                    inp = np.zeros(self.snn.layers[0].batch_input_shape,
                                   'float32')
                    for sample_idx in range(s['batch_size']):
                        # Buffer event sequence because we will be removing
                        # elements from original list:
                        xaddr_sample = list(x_batch_xaddr[sample_idx])
                        yaddr_sample = list(x_batch_yaddr[sample_idx])
                        ts_sample = list(x_batch_ts[sample_idx])
                        first_ts_of_frame = ts_sample[0] if ts_sample else 0
                        for x, y, ts in zip(xaddr_sample, yaddr_sample,
                                            ts_sample):
                            if inp[sample_idx, 0, y, x] == 0:
                                inp[sample_idx, 0, y, x] = 1
                                # Can't use .popleft()
                                x_batch_xaddr[sample_idx].remove(x)
                                x_batch_yaddr[sample_idx].remove(y)
                                x_batch_ts[sample_idx].remove(ts)
                            if ts - first_ts_of_frame > 1000:
                                break
                # Main step: Propagate input through network and record output
                # spikes.
                self.set_time(sim_step)
                out_spikes = self.snn.predict_on_batch(inp)
                if remove_classifier:
                    output += np.argmax(np.reshape(out_spikes.astype('int32'),
                                                   (out_spikes.shape[0], -1)),
                                        axis=1)
                else:
                    output += out_spikes.astype('int32')
                # Get result by comparing the guessed class (i.e. the index
                # of the neuron in the last layer which spiked most) to the
                # ground truth.
                guesses_batch = np.argmax(output, axis=1)
                # Find sample indices for which there was no output spike yet
                undecided = np.where(np.sum(output != 0, axis=1) == 0)
                # Assign negative value such that undecided samples count as
                # wrongly classified.
                guesses_batch[undecided] = -1
                top1err_vs_time.append(np.around(
                    np.mean(truth_batch != guesses_batch), 4))
                # Record the spiketrains of each neuron in each layer.
                j = 0
                if 'spikecounts' in s['plot_vars'] + s['log_vars']:
                    total_spikecount = np.zeros(s['batch_size'])
                for layer in self.snn.layers:
                    if not hasattr(layer, 'spiketrain'):
                        continue
                    if 'spiketrains' in s['log_vars']:
                        spiketrains_batch[j][0][Ellipsis, sim_step_int] = \
                            layer.spiketrain.get_value()
                    if 'spikecounts' in s['plot_vars'] + s['log_vars']:
                        layer_spikecounts[:, j, sim_step_int] = \
                            layer.total_spikecount.get_value()
                        total_spikecount += \
                            layer_spikecounts[:, j, sim_step_int]
                    j += 1
                if 'spiketrains' in s['log_vars'] and False:
                    self.monitor_debug_vars(sim_step_int, inp)
                if 'spikecounts' in s['plot_vars'] + s['log_vars']:
                    total_spikecount_t[sim_step_int] = total_spikecount
                if 'input_t' in s['log_vars']:
                    input_t[sim_step_int] = inp
                sim_step_int += 1
                if s['verbose'] > 0 and sim_step % 1 == 0:
                    echo('{:.2%}_'.format(1-top1err_vs_time[-1]))
            if 'spiketrains' in s['log_vars'] and False:
                self.save_debug_vars(log_dir, spiketrains_batch)

            truth += list(truth_batch)
            guesses += list(guesses_batch)
            top1acc_moving = np.mean(np.array(truth) == np.array(guesses))
            top5score_moving += get_top5score(truth_batch, output)
            top5acc_moving = top5score_moving / ((batch_idx + 1) *
                                                 s['batch_size'])
            net_top1err_t.append(top1err_vs_time)
            with open(path_pred, 'a') as f_pred:
                f_pred.write("Predictions of batch {}/{}: {}\n".format(
                    batch_idx + 1, num_batches, str(guesses)))
            with open(path_target, 'a') as f_target:
                f_target.write("True classes of batch {}/{}: {}\n".format(
                    batch_idx + 1, num_batches, str(truth)))
            with open(path_acc, 'a') as f_acc:
                f_acc.write(
                    "Moving average accuracy after batch {}/{}: {:.2%}."
                    "\n".format(batch_idx + 1, num_batches, top1acc_moving))
            with open(path_top5acc, 'a') as f_top5acc:
                f_top5acc.write(
                    "Moving average of top-5-accuracy after batch {}/{}: "
                    "{:.2%}.\n".format(batch_idx + 1, num_batches,
                                       top5acc_moving))
            if s['verbose'] > 0:
                print("\nBatch {} of {} completed ({:.1%})".format(
                    batch_idx + 1, num_batches, (batch_idx + 1) / num_batches))
                print("Moving average accuracy: {:.2%}.\n".format(
                    top1acc_moving))
                print("Moving top-5 accuracy: {:.2%}.\n".format(top5acc_moving))
            if 'activations' in s['log_vars'] + s['plot_vars']:
                print("Calculating activations...")
                activations_batch = get_activations_batch(self.parsed_model,
                                                          x_batch)
            if 'input_image' in s['plot_vars'] and x_batch is not None:
                plot_input_image(x_batch[0], int(truth_batch[0]), log_dir)
            if 'input_t' in s['log_vars']:
                print("Saving batch input vs time...")
                np.savez_compressed(os.path.join(path_input, str(batch_idx)),
                                    input_t=input_t)
            if 'error_t' in s['plot_vars']:
                ann_err = self.ANN_err if hasattr(self, 'ANN_err') else None
                plot_error_vs_time(top1err_vs_time, ann_err, log_dir,
                                   s['batch_size'])
            if 'spikecounts' in s['plot_vars']:
                plot_spikecount_vs_time(total_spikecount_t, log_dir)
            if 'confusion_matrix' in s['plot_vars']:
                plot_confusion_matrix(truth, guesses, log_dir,
                                      list(np.arange(num_classes)))
            if 'activations' in s['log_vars']:
                print("Saving batch activations...")
                np.savez_compressed(os.path.join(path_activ, str(batch_idx)),
                                    activations=activations_batch)
            if 'spiketrains' in s['log_vars']:
                print("Saving batch spiketrains...")
                np.savez_compressed(os.path.join(path_trains, str(batch_idx)),
                                    spiketrains=spiketrains_batch)
            if 'spikecounts' in s['log_vars']:
                print("Saving batch spikecounts...")
                np.savez_compressed(os.path.join(path_count, str(batch_idx)),
                                    spikecounts=total_spikecount_t)
                np.savez_compressed(os.path.join(path_count,
                                                 str(batch_idx) + '_l'),
                                    layer_spike_counts=layer_spikecounts)
            if any(v in s['plot_vars'] for v in
                   ['activations', 'spiketrains', 'spikerates',
                    'correlation', 'hist_spikerates_activations']):
                output_graphs(spiketrains_batch, activations_batch, log_dir, 0)
        count = np.zeros(num_classes)
        match = np.zeros(num_classes)
        for gt, p in zip(truth, guesses):
            count[gt] += 1
            if gt == p:
                match[gt] += 1
        avg_acc = np.mean(match / count)
        top1acc_total = np.mean(np.array(truth) == np.array(guesses))
        np.savez_compressed(os.path.join(log_dir, 'net_top1err_t'),
                            net_top1err_t=np.array(net_top1err_t))
        if 'confusion_matrix' in s['plot_vars']:
            plot_confusion_matrix(truth, guesses, log_dir,
                                  list(np.arange(num_classes)))
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

    def init_debug_vars(self):
        """Initialize debug variables."""

        t = int(settings['duration'] / settings['dt'])
        self.debug_vars = {
            'mem4': np.empty(t),
            'mem8': np.empty(t),
            'mem14': np.empty(t),
            'inp_t': np.zeros(t),
            # 'mem_l_t': np.empty((14, t)),
            # 'spikes_l_t': np.zeros((14, t))
        }

    def monitor_debug_vars(self, t_idx, inp):
        """Monitor debug variables.

        Parameters
        ----------

        t_idx: int
        inp:
        """

        self.debug_vars['mem4'][t_idx] = \
            self.snn.layers[5].mem.get_value()[0, 6, 3, 30]
        self.debug_vars['mem8'][t_idx] = \
            self.snn.layers[9].mem.get_value()[0, 0, 0, 0]
        self.debug_vars['mem14'][t_idx] = \
            self.snn.layers[15].mem.get_value()[0, 4, 0, 34]
        self.debug_vars['inp_t'][t_idx] = inp[0, 0, 32, 32]
        # k = -1
        # self.debug_vars['mem_l_t'][0][t_idx] = \
        #     self.snn.layers[1].mem.get_value()[k, 0, 30, 30]
        # self.debug_vars['mem_l_t'][1][t_idx] = \
        #     self.snn.layers[2].mem.get_value()[k, 0, 15, 15]
        # self.debug_vars['mem_l_t'][2][t_idx] = \
        #     self.snn.layers[3].mem.get_value()[k, 0, 14, 14]
        # self.debug_vars['mem_l_t'][3][t_idx] = \
        #     self.snn.layers[4].mem.get_value()[k, 0, 7, 7]
        # self.debug_vars['mem_l_t'][4][t_idx] = \
        #     self.snn.layers[5].mem.get_value()[k, 0, 6, 6]
        # self.debug_vars['mem_l_t'][5][t_idx] = \
        #     self.snn.layers[6].mem.get_value()[k, 0, 3, 3]
        # self.debug_vars['mem_l_t'][6][t_idx] = \
        #     self.snn.layers[7].mem.get_value()[k, 0, 2, 2]
        # self.debug_vars['mem_l_t'][7][t_idx] = \
        #     self.snn.layers[8].mem.get_value()[k, 0, 1, 1]
        # self.debug_vars['mem_l_t'][8][t_idx] = \
        #     self.snn.layers[9].mem.get_value()[k, 0, 1, 1]
        # self.debug_vars['mem_l_t'][9][t_idx] = \
        #     self.snn.layers[10].mem.get_value()[k, 0, 0, 0]
        # self.debug_vars['mem_l_t'][10][t_idx] = \
        #     self.snn.layers[12].mem.get_value()[k, 0]  # All 4 output classes
        # self.debug_vars['mem_l_t'][11][t_idx] = \
        #     self.snn.layers[12].mem.get_value()[k, 1]
        # self.debug_vars['mem_l_t'][12][t_idx] = \
        #     self.snn.layers[12].mem.get_value()[k, 2]
        # self.debug_vars['mem_l_t'][13][t_idx] = \
        #     self.snn.layers[12].mem.get_value()[k, 3]
        # self.debug_vars['spikes_l_t'][0][t_idx] = \
        #     self.snn.layers[1].spiketrain.get_value()[k, 0, 30, 30]
        # self.debug_vars['spikes_l_t'][1][t_idx] = \
        #     self.snn.layers[2].spiketrain.get_value()[k, 0, 15, 15]
        # self.debug_vars['spikes_l_t'][2][t_idx] = \
        #     self.snn.layers[3].spiketrain.get_value()[k, 0, 14, 14]
        # self.debug_vars['spikes_l_t'][3][t_idx] = \
        #     self.snn.layers[4].spiketrain.get_value()[k, 0, 7, 7]
        # self.debug_vars['spikes_l_t'][4][t_idx] = \
        #     self.snn.layers[5].spiketrain.get_value()[k, 0, 6, 6]
        # self.debug_vars['spikes_l_t'][5][t_idx] = \
        #     self.snn.layers[6].spiketrain.get_value()[k, 0, 3, 3]
        # self.debug_vars['spikes_l_t'][6][t_idx] = \
        #     self.snn.layers[7].spiketrain.get_value()[k, 0, 2, 2]
        # self.debug_vars['spikes_l_t'][7][t_idx] = \
        #     self.snn.layers[8].spiketrain.get_value()[k, 0, 1, 1]
        # self.debug_vars['spikes_l_t'][8][t_idx] = \
        #     self.snn.layers[9].spiketrain.get_value()[k, 0, 1, 1]
        # self.debug_vars['spikes_l_t'][9][t_idx] = \
        #     self.snn.layers[10].spiketrain.get_value()[k, 0, 0, 0]
        # self.debug_vars['spikes_l_t'][10][t_idx] = \
        #     self.snn.layers[12].spiketrain.get_value()[k, 0]
        # self.debug_vars['spikes_l_t'][11][t_idx] = \
        #     self.snn.layers[12].spiketrain.get_value()[k, 1]
        # self.debug_vars['spikes_l_t'][12][t_idx] = \
        #     self.snn.layers[12].spiketrain.get_value()[k, 2]
        # self.debug_vars['spikes_l_t'][13][t_idx] = \
        #     self.snn.layers[12].spiketrain.get_value()[k, 3]

    def save_debug_vars(self, path, spiketrains_batch):
        """Save debug variables.

        Parameters
        ----------

        spiketrains_batch :
        path: str
            Destination path.
        """

        for sb, label in spiketrains_batch:
            if '03AveragePooling2D_64' in label:
                self.debug_vars['inputspikes4'] = sb[0, :, 3, 30, :]
            if '06AveragePooling2D_192' in label:
                self.debug_vars['inputspikes8'] = sb[0, :, 0, 0, :]
            if '13AveragePooling2D_192' in label:
                self.debug_vars['inputspikes14'] = sb[0, :, 0, 34, :]

        self.debug_vars['weights4'] = \
            self.snn.layers[5].get_weights()[0][6, :, 0, 0]
        self.debug_vars['bias4'] = self.snn.layers[5].get_weights()[1][6]
        self.debug_vars['weights8'] = \
            self.snn.layers[9].get_weights()[0][0, :, 0, 0]
        self.debug_vars['bias8'] = self.snn.layers[9].get_weights()[1][0]
        self.debug_vars['weights14'] = \
            self.snn.layers[15].get_weights()[0][4, :, 0, 0]
        self.debug_vars['bias14'] = self.snn.layers[15].get_weights()[1][4]

        np.savez_compressed(os.path.join(path, 'debug_vars'), **self.debug_vars)


def remove_outliers(timestamps, xaddr, yaddr, pol, x_max=240, y_max=180):
    """Remove outliers from DVS data.

    Parameters
    ----------
    timestamps :
    xaddr :
    yaddr :
    pol :
    x_max :
    y_max :

    Returns
    -------

    """

    len_orig = len(timestamps)
    xaddr_valid = np.where(np.array(xaddr) < x_max)
    yaddr_valid = np.where(np.array(yaddr) < y_max)
    xy_valid = np.intersect1d(xaddr_valid[0], yaddr_valid[0], True)
    xaddr = np.array(xaddr)[xy_valid]
    yaddr = np.array(yaddr)[xy_valid]
    timestamps = np.array(timestamps)[xy_valid]
    pol = np.array(pol)[xy_valid]
    num_outliers = len_orig - len(timestamps)
    if num_outliers:
        print("Removed {} outliers.".format(num_outliers))
    return timestamps, xaddr, yaddr, pol


def load_dvs_sequence(filename, xyrange=None):
    """

    Parameters
    ----------

    filename:
    xyrange:

    Returns
    -------

    """

    from snntoolbox.io_utils.AedatTools import ImportAedat

    print("Loading DVS sample {}...".format(filename))
    events = ImportAedat.import_aedat({'filePathAndName':
                                       filename})['data']['polarity']
    timestamps = events['timeStamp']
    xaddr = events['x']
    yaddr = events['y']
    pol = events['polarity']

    # Remove events with addresses outside valid range
    if xyrange:
        timestamps, xaddr, yaddr, pol = remove_outliers(
            timestamps, xaddr, yaddr, pol, xyrange[0], xyrange[1])

    return xaddr, yaddr, timestamps


class DVSIterator(object):
    """

    Parameters
    ----------
    dataset_path :
    batch_size :
    scale:

    Returns
    -------

    """

    def __init__(self, dataset_path, batch_size, label_dict=None,
                 scale=None, num_events_per_sample=1000):
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.batch_idx = 0
        self.scale = scale
        self.xaddr_sequence = None
        self.yaddr_sequence = None
        self.dvs_sample = None
        self.num_events_of_sample = 0
        self.dvs_sample_idx = -1
        self.num_events_per_sample = num_events_per_sample
        self.num_events_per_batch = batch_size * num_events_per_sample

        # Count the number of samples and classes
        classes = [subdir for subdir in sorted(os.listdir(dataset_path))
                   if os.path.isdir(os.path.join(dataset_path, subdir))]

        self.label_dict = dict(zip(classes, range(len(classes)))) \
            if not label_dict else label_dict
        self.num_classes = len(label_dict)
        assert self.num_classes == len(classes), \
            "The number of classes provided by label_dict {} does not match " \
            "the number of subdirectories found in dataset_path {}.".format(
                self.label_dict, self.dataset_path)

        self.filenames = []
        labels = []
        self.num_samples = 0
        for subdir in classes:
            for fname in sorted(os.listdir(os.path.join(dataset_path, subdir))):
                is_valid = False
                for extension in {'aedat'}:
                    if fname.lower().endswith('.' + extension):
                        is_valid = True
                        break
                if is_valid:
                    labels.append(self.label_dict[subdir])
                    self.filenames.append(os.path.join(subdir, fname))
                    self.num_samples += 1
        self.labels = np.array(labels, 'int32')
        print("Found {} samples belonging to {} classes.".format(
            self.num_samples, self.num_classes))

    def __next__(self):
        from snntoolbox.io_utils.common import to_categorical
        from collections import deque

        while self.num_events_per_batch * (self.batch_idx + 1) >= \
                self.num_events_of_sample:
            self.dvs_sample_idx += 1
            if self.dvs_sample_idx == len(self.filenames):
                raise StopIteration()
            filepath = os.path.join(self.dataset_path,
                                    self.filenames[self.dvs_sample_idx])
            self.dvs_sample = load_dvs_sequence(filepath, (240, 180))
            self.num_events_of_sample = len(self.dvs_sample[0])
            self.batch_idx = 0
            print("Total number of events of this sample: {}.".format(
                self.num_events_of_sample))
            print("Number of batches: {:d}.".format(
                int(self.num_events_of_sample / self.num_events_per_batch)))

        print("Extracting batch of samples à {} events from DVS sequence..."
              "".format(self.num_events_per_sample))
        x_batch_xaddr = [deque() for _ in range(self.batch_size)]
        x_batch_yaddr = [deque() for _ in range(self.batch_size)]
        x_batch_ts = [deque() for _ in range(self.batch_size)]
        for sample_idx in range(self.batch_size):
            start_event = self.num_events_per_batch * self.batch_idx + \
                          self.num_events_per_sample * sample_idx
            event_idxs = range(start_event,
                               start_event + self.num_events_per_sample)
            event_sums = np.zeros((64, 64), 'int32')
            xaddr_sub = []
            yaddr_sub = []
            for x, y in zip(self.dvs_sample[0][event_idxs],
                            self.dvs_sample[1][event_idxs]):
                if self.scale:
                    # Subsample from 240x180 to e.g. 64x64
                    x = int(x / self.scale[0])
                    y = int(y / self.scale[1])
                event_sums[y, x] += 1
                xaddr_sub.append(x)
                yaddr_sub.append(y)
            sigma = np.std(event_sums)
            # Clip number of events per pixel to three-sigma
            np.clip(event_sums, 0, 3*sigma, event_sums)
            print("Discarded {} events during 3-sigma standardization.".format(
                self.num_events_per_sample - np.sum(event_sums)))
            ts_sample = self.dvs_sample[2][event_idxs]
            for x, y, ts in zip(xaddr_sub, yaddr_sub, ts_sample):
                if event_sums[y, x] > 0:
                    x_batch_xaddr[sample_idx].append(x)
                    x_batch_yaddr[sample_idx].append(y)
                    x_batch_ts[sample_idx].append(ts)
                    event_sums[y, x] -= 1

        # Each sample in the batch has the same label because it is generated
        # from the same DVS sequence.
        y_batch = np.broadcast_to(to_categorical(
            [self.labels[self.dvs_sample_idx]], self.num_classes),
            (self.batch_size, self.num_classes))

        self.batch_idx += 1

        return x_batch_xaddr, x_batch_yaddr, x_batch_ts, y_batch
