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
from snntoolbox.core.util import in_top_k

standard_library.install_aliases()


remove_classifier = False


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
        # Logging variables
        self.spiketrains_n_b_l_t = self.activations_n_b_l = None
        self.input_b_l_t = self.mem_n_b_l_t = None
        self.top1err_b_t = self.top5err_b_t = None
        self.operations_b_t = self.ann_ops = None
        self.num_neurons = self.num_neurons_with_bias = None
        self.fanin = self.fanout = None
        # ``rescale_fac`` globally scales spike probability when using Poisson
        # input.
        self.rescale_fac = 1
        self.num_classes = 0
        self.top_k = 5

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
            if len(inbound) == 1:
                inbound = inbound[0]
            spiking_layers[layer.name] = \
                spike_layer.from_config(layer.get_config())(inbound)

        if verbose:
            print("Compiling spiking model...\n")
        self.snn = keras.models.Model(
            input_images, spiking_layers[parsed_model.layers[-1].name])
        self.snn.compile('sgd', 'categorical_crossentropy',
                         ['accuracy'])
        self.snn.set_weights(parsed_model.get_weights())
        for layer in self.snn.layers:
            if hasattr(layer, 'b'):
                # Adjust biases to time resolution of simulator.
                layer.b.set_value(layer.b.get_value() * settings['dt'])
                if bias_relaxation:  # Experimental
                    layer.b0.set_value(layer.b.get_value())

        if self.fanin is None:
            from snntoolbox.core.util import get_ann_ops
            num_neurons, num_neurons_with_bias, fanin = self.set_connectivity()
            self.ann_ops = get_ann_ops(num_neurons, num_neurons_with_bias,
                                       fanin)
            print("Number of operations of ANN: {}".format(self.ann_ops))

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
        from snntoolbox.core.util import get_activations_batch
        from snntoolbox.core.util import echo, get_layer_ops
        from snntoolbox.io_utils.plotting import output_graphs
        from snntoolbox.io_utils.plotting import plot_confusion_matrix
        from snntoolbox.io_utils.plotting import plot_error_vs_time
        from snntoolbox.io_utils.plotting import plot_input_image
        from snntoolbox.io_utils.plotting import plot_ops_vs_time
        from snntoolbox.io_utils.AedatTools.DVSIterator import DVSIterator

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

        # Extract certain samples from test set, if user specified such a list.
        si = s['sample_indices_to_test'].copy() \
            if 'sample_indices_to_test' in s else []
        if not si == []:
            if dataflow is not None:
                batch_idx = 0
                x_test = []
                y_test = []
                target_idx = si.pop(0)
                while len(x_test) < s['num_to_test']:
                    x_b_l, y_b = dataflow.next()
                    for i in range(s['batch_size']):
                        if batch_idx * s['batch_size'] + i == target_idx:
                            x_test.append(x_b_l[i])
                            y_test.append(y_b[i])
                            if len(si) > 0:
                                target_idx = si.pop(0)
                    batch_idx += 1
                x_test = np.array(x_test)
                y_test = np.array(y_test)
            else:
                x_test = np.array([x_test[i] for i in si])
                y_test = np.array([y_test[i] for i in si])

        # Divide the test set into batches and run all samples in a batch in
        # parallel.
        num_batches = int(1e9) if s['dataset_format'] == 'aedat' else \
            int(np.floor(s['num_to_test'] / s['batch_size']))

        top5score_moving = 0
        score_ann = np.zeros(2)
        truth_d = []  # Filled up with correct classes of all test samples.
        guesses_d = []  # Filled up with guessed classes of all test samples.
        guesses_b = np.zeros(s['batch_size'])  # Guesses of one batch.
        x_b_xaddr = x_b_yaddr = x_b_ts = None
        x_b_l = y_b = None
        dvs_gen = None
        if s['dataset_format'] == 'aedat':
            dvs_gen = DVSIterator(s['dataset_path'], s['batch_size'],
                                  s['label_dict'], s['subsample_facs'],
                                  s['num_dvs_events_per_sample'])

        # Prepare files to write moving accuracy and error to.
        path_log_vars = os.path.join(log_dir, 'log_vars')
        if not os.path.isdir(path_log_vars):
            os.makedirs(path_log_vars)
        path_acc = os.path.join(log_dir, 'accuracy.txt')
        if os.path.isfile(path_acc):
            os.remove(path_acc)

        self.init_log_vars()
        self.num_classes = int(self.snn.layers[-1].output_shape[-1])
        self.top_k = min(self.num_classes, 5)

        for batch_idx in range(num_batches):
            # Get a batch of samples
            if dataflow is not None and len(s['sample_indices_to_test']) == 0:
                x_b_l, y_b = dataflow.next()
            elif not s['dataset_format'] == 'aedat':
                batch_idxs = range(s['batch_size'] * batch_idx,
                                   s['batch_size'] * (batch_idx + 1))
                x_b_l = x_test[batch_idxs, :]
                y_b = y_test[batch_idxs, :]
            if s['dataset_format'] == 'aedat':
                try:
                    x_b_xaddr, x_b_yaddr, x_b_ts, y_b = dvs_gen.__next__()
                except StopIteration:
                    break
                if any({'activations', 'correlation',
                        'hist_spikerates_activations'} & s['plot_vars']) or \
                        'activations_n_b_l' in s['log_vars']:
                    if x_test is None:
                        from snntoolbox.io_utils.common import load_dataset
                        dataset_path_npz = '/home/rbodo/.snntoolbox/Datasets/' \
                                           'roshambo/frames_background'
                        x_test = load_dataset(dataset_path_npz, 'x_test.npz')
                    batch_idxs = range(s['batch_size'] * batch_idx,
                                       s['batch_size'] * (batch_idx + 1))
                    x_b_l = x_test[batch_idxs, :]
            truth_b = np.argmax(y_b, axis=1)

            # Either use Poisson spiketrains as inputs to the SNN, or take the
            # original data.
            if s['poisson_input']:
                # This factor determines the probability threshold for cells in
                # the input layer to fire a spike. Increasing ``input_rate``
                # increases the firing rate of the input and subsequent layers.
                self.rescale_fac = np.max(x_b_l)*1000/s['input_rate']/s['dt']
            elif s['dataset_format'] == 'aedat':
                pass
            else:
                # Simply use the analog values of the original data as input.
                input_b_l = x_b_l * s['dt']

            # Reset network variables.
            self.reset(batch_idx)

            # Allocate variables to monitor during simulation
            output_b_l = np.zeros((s['batch_size'], self.num_classes), 'int32')

            input_spikecount = 0
            sim_step_int = 0
            print("Starting new simulation...\n")
            # Loop through simulation time.
            for sim_step in range(s['dt'], s['duration']+s['dt'], s['dt']):
                # Generate input, in case it changes with each simulation step:
                if s['poisson_input']:
                    if input_spikecount < s['num_poisson_events_per_sample'] \
                            or s['num_poisson_events_per_sample'] < 0:
                        spike_snapshot = np.random.random_sample(x_b_l.shape) \
                                         * self.rescale_fac
                        input_b_l = (spike_snapshot <= np.abs(x_b_l)).astype(
                            'float32')
                        input_spikecount += \
                            np.count_nonzero(input_b_l) / s['batch_size']
                        # For BinaryNets, with input that is not normalized and
                        # not all positive, we stimulate with spikes of the same
                        # size as the maximum activation, and the same sign as
                        # the corresponding activation. Is there a better
                        # solution?
                        input_b_l *= np.max(x_b_l) * np.sign(x_b_l)
                    else:
                        input_b_l = np.zeros(x_b_l.shape)
                elif s['dataset_format'] == 'aedat':
                    input_b_l = np.zeros(self.snn.layers[0].batch_input_shape,
                                         'float32')
                    for sample_idx in range(s['batch_size']):
                        # Buffer event sequence because we will be removing
                        # elements from original list:
                        xaddr_sample = list(x_b_xaddr[sample_idx])
                        yaddr_sample = list(x_b_yaddr[sample_idx])
                        ts_sample = list(x_b_ts[sample_idx])
                        first_ts_of_frame = ts_sample[0] if ts_sample else 0
                        for x, y, ts in zip(xaddr_sample, yaddr_sample,
                                            ts_sample):
                            if input_b_l[sample_idx, 0, y, x] == 0:
                                input_b_l[sample_idx, 0, y, x] = 1
                                # Can't use .popleft()
                                x_b_xaddr[sample_idx].remove(x)
                                x_b_yaddr[sample_idx].remove(y)
                                x_b_ts[sample_idx].remove(ts)
                            if ts - first_ts_of_frame > s['eventframe_width']:
                                break
                # Main step: Propagate input through network and record output
                # spikes.
                self.set_time(sim_step)
                out_spikes = self.snn.predict_on_batch(input_b_l)
                if remove_classifier:
                    output_b_l += np.argmax(np.reshape(
                        out_spikes.astype('int32'), (out_spikes.shape[0], -1)),
                        axis=1)
                else:
                    output_b_l += out_spikes.astype('int32')
                # Get result by comparing the guessed class (i.e. the index
                # of the neuron in the last layer which spiked most) to the
                # ground truth.
                guesses_b = np.argmax(output_b_l, axis=1)
                # Find sample indices for which there was no output spike yet
                undecided = np.where(np.sum(output_b_l != 0, axis=1) == 0)
                # Assign negative value such that undecided samples count as
                # wrongly classified.
                guesses_b[undecided] = -1
                self.top1err_b_t[:, sim_step_int] = truth_b != guesses_b
                self.top5err_b_t[:, sim_step_int] = \
                    ~in_top_k(output_b_l, truth_b, self.top_k)
                # Record neuron variables.
                i = j = 0
                for layer in self.snn.layers:
                    if hasattr(layer, 'spiketrain'):
                        if self.spiketrains_n_b_l_t is not None:
                            self.spiketrains_n_b_l_t[i][0][Ellipsis,
                                                           sim_step_int] = \
                                layer.spiketrain.get_value()
                        if self.operations_b_t is not None:
                            self.operations_b_t[:, sim_step_int] += \
                                get_layer_ops(layer.spiketrain.get_value(),
                                              self.fanout[i+1],
                                              self.num_neurons_with_bias[i+1])
                        i += 1
                    if hasattr(layer, 'mem') and self.mem_n_b_l_t is not None:
                        self.mem_n_b_l_t[j][0][Ellipsis, sim_step_int] = \
                            layer.mem.get_value()
                        j += 1
                if 'input_b_l_t' in s['log_vars']:
                    self.input_b_l_t[Ellipsis, sim_step_int] = input_b_l
                if self.operations_b_t is not None:
                    if s['poisson_input'] or s['dataset_format'] == 'aedat':
                        input_ops = get_layer_ops(input_b_l, self.fanout[0])
                    else:
                        input_ops = np.ones((s['batch_size'])) * \
                            self.num_neurons[1]
                        if sim_step_int == 0:
                            input_ops *= 2 * self.fanin[1]  # MACs for convol.
                    self.operations_b_t[:, sim_step_int] += input_ops
                top1err = np.around(np.mean(self.top1err_b_t[:, sim_step_int]),
                                    4)
                sim_step_int += 1
                if s['verbose'] > 0 and sim_step % 1 == 0:
                    echo('{:.2%}_'.format(1-top1err))

            num_samples_seen = (batch_idx + 1) * s['batch_size']
            truth_d += list(truth_b)
            guesses_d += list(guesses_b)
            top1acc_moving = np.mean(np.array(truth_d) == np.array(guesses_d))
            top5score_moving += sum(in_top_k(output_b_l, truth_b, self.top_k))
            top5acc_moving = top5score_moving / num_samples_seen
            if s['verbose'] > 0:
                print("\nBatch {} of {} completed ({:.1%})".format(
                    batch_idx + 1, num_batches, (batch_idx + 1) / num_batches))
                print("Moving accuracy of SNN (top-1, top-5): {:.2%}, {:.2%}."
                      "".format(top1acc_moving, top5acc_moving))
            with open(path_acc, 'a') as f_acc:
                f_acc.write("{} {:.2%} {:.2%}\n".format(
                    num_samples_seen, top1acc_moving, top5acc_moving))

            # Evaluate ANN
            score = self.parsed_model.test_on_batch(x_b_l, y_b)
            score_ann += score[1:]
            top1acc_moving_ann, top5acc_moving_ann = score_ann / num_samples_seen
            print("Moving accuracy of ANN (top-1, top-5): {:.2%}, {:.2%}."
                  "\n".format(top1acc_moving_ann, top5acc_moving_ann))

            if 'input_image' in s['plot_vars'] and x_b_l is not None:
                plot_input_image(x_b_l[0], int(truth_b[0]), log_dir)
            if 'error_t' in s['plot_vars']:
                ann_err = self.ANN_err if hasattr(self, 'ANN_err') else None
                plot_error_vs_time(self.top1err_b_t, self.top5err_b_t,
                                   top1acc_moving_ann, top5acc_moving_ann,
                                   log_dir)
            if 'confusion_matrix' in s['plot_vars']:
                plot_confusion_matrix(truth_d, guesses_d, log_dir,
                                      list(np.arange(self.num_classes)))
            # Cumulate operation count over time and scale to MOps.
            if self.operations_b_t is not None:
                np.cumsum(np.divide(self.operations_b_t, 1e6), 1,
                          out=self.operations_b_t)
            if 'operations' in s['plot_vars']:
                plot_ops_vs_time(self.operations_b_t, log_dir)
            if any({'activations', 'correlation', 'hist_spikerates_activations'}
                   & s['plot_vars']) or 'activations_n_b_l' in s['log_vars']:
                print("Calculating activations...")
                self.activations_n_b_l = get_activations_batch(
                    self.parsed_model, x_b_l)
            log_vars = {key: getattr(self, key) for key in s['log_vars']}
            log_vars['top1err_b_t'] = self.top1err_b_t
            log_vars['top5err_b_t'] = self.top5err_b_t
            np.savez_compressed(os.path.join(path_log_vars, str(batch_idx)),
                                **log_vars)
            plot_vars = {}
            if any({'activations', 'correlation',
                    'hist_spikerates_activations'} & s['plot_vars']):
                plot_vars['activations_n_b_l'] = self.activations_n_b_l
            if any({'spiketrains', 'spikerates', 'correlation', 'spikecounts',
                    'hist_spikerates_activations'} & s['plot_vars']):
                plot_vars['spiketrains_n_b_l_t'] = self.spiketrains_n_b_l_t
            output_graphs(plot_vars, log_dir, 0)
        # Compute average accuracy, taking into account number of samples per
        # class
        count = np.zeros(self.num_classes)
        match = np.zeros(self.num_classes)
        for gt, p in zip(truth_d, guesses_d):
            count[gt] += 1
            if gt == p:
                match[gt] += 1
        avg_acc = np.mean(match / count)
        top1acc_total = np.mean(np.array(truth_d) == np.array(guesses_d))
        if 'confusion_matrix' in s['plot_vars']:
            plot_confusion_matrix(truth_d, guesses_d, log_dir,
                                  list(np.arange(self.num_classes)))
        print("Simulation finished.\n\n")
        print("Total accuracy: {:.2%} on {} test samples.\n\n".format(
            top1acc_total, len(guesses_d)))
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

    def reset(self, sample_idx):
        """Reset network variables."""

        for layer in self.snn.layers[1:]:  # Skip input layer
            layer.reset(sample_idx)

    def init_log_vars(self):
        """Initialize debug variables."""

        num_timesteps = int(settings['duration'] / settings['dt'])

        if 'input_b_l_t' in settings['log_vars']:
            self.input_b_l_t = np.empty(
                list(self.snn.input_shape) + [num_timesteps], np.bool)

        if any({'spiketrains', 'spikerates', 'correlation', 'spikecounts',
                'hist_spikerates_activations'} & settings['plot_vars']) \
                or 'spiketrains_n_b_l_t' in settings['log_vars']:
            self.spiketrains_n_b_l_t = []
            for layer in self.snn.layers:
                if not hasattr(layer, 'spiketrain'):
                    continue
                shape = list(layer.output_shape) + [num_timesteps]
                self.spiketrains_n_b_l_t.append((np.zeros(shape, 'float32'),
                                                 layer.name))

        if 'operations' in settings['plot_vars'] or \
                'operations_b_t' in settings['log_vars']:
            self.operations_b_t = np.zeros((settings['batch_size'],
                                            num_timesteps))

        if 'mem_n_b_l_t' in settings['log_vars'] \
                or 'mem' in settings['plot_vars']:
            self.mem_n_b_l_t = []
            for layer in self.snn.layers:
                if not hasattr(layer, 'mem'):
                    continue
                shape = list(layer.output_shape) + [num_timesteps]
                self.mem_n_b_l_t.append((np.zeros(shape, 'float32'),
                                         layer.name))

        self.top1err_b_t = np.empty((settings['batch_size'], num_timesteps),
                                    np.bool)
        self.top5err_b_t = np.empty((settings['batch_size'], num_timesteps),
                                    np.bool)

    def set_connectivity(self):
        """
        Set connectivity statistics needed to compute the number of operations
        in the network, e.g. fanin, fanout, number_of_neurons,
        number_of_neurons_with_bias.
        """

        from snntoolbox.core.util import get_fanin, get_fanout

        self.fanin = [0]
        self.fanout = [int(np.multiply(np.prod(self.snn.layers[1].kernel_size),
                                       self.snn.layers[1].filters))]
        self.num_neurons = [np.product(self.snn.input_shape[1:])]
        self.num_neurons_with_bias = [0]

        for layer in self.snn.layers:
            if hasattr(layer, 'spiketrain'):
                self.fanin.append(get_fanin(layer))
                self.fanout.append(get_fanout(layer))
                self.num_neurons.append(np.prod(layer.output_shape[1:]))
                if hasattr(layer, 'b') and any(layer.b.get_value()):
                    print("Detected layer with biases: {}".format(layer.name))
                    self.num_neurons_with_bias.append(self.num_neurons[-1])
                else:
                    self.num_neurons_with_bias.append(0)

        return self.num_neurons, self.num_neurons_with_bias, self.fanin
