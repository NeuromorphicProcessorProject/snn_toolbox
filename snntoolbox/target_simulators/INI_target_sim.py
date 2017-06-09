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
import keras
from future import standard_library
from snntoolbox.config import initialize_simulator
from snntoolbox.core.inisim import bias_relaxation
from snntoolbox.core.util import in_top_k, get_plot_keys, get_log_keys

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

    def __init__(self, config, queue=None):
        """Init function."""

        self.config = config
        self.queue = queue
        self.sim = initialize_simulator(config['simulation']['simulator'])
        self.snn = None
        self.parsed_model = None
        # Logging variables
        self.spiketrains_n_b_l_t = self.activations_n_b_l = None
        self.input_b_l_t = self.mem_n_b_l_t = None
        self.top1err_b_t = self.top5err_b_t = None
        self.operations_b_t = self.operations_ann = None
        self.top1err_ann = self.top5err_ann = None
        self.num_neurons = self.num_neurons_with_bias = None
        self.fanin = self.fanout = None
        # ``rescale_fac`` globally scales spike probability when using Poisson
        # input.
        self.rescale_fac = 1
        self.num_classes = 0
        self.top_k = 5

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
            Parsed input model.
        """

        print("Building spiking model...")

        self.parsed_model = parsed_model

        if 'batch_size' in kwargs:
            batch_shape = [kwargs[str('batch_size')]] + \
                          list(parsed_model.layers[0].batch_input_shape)[1:]
        else:
            batch_shape = list(parsed_model.layers[0].batch_input_shape)
        if batch_shape[0] is None:
            batch_shape[0] = self.config.getint('simulation', 'batch_size')

        input_images = keras.layers.Input(batch_shape=batch_shape)
        spiking_layers = {parsed_model.layers[0].name: input_images}

        # Iterate over layers to create spiking neurons and connections.
        binary_activation = None
        for layer in parsed_model.layers[1:]:  # Skip input layer
            print("Building layer: {}".format(layer.name))
            spike_layer = getattr(self.sim, 'Spike' + layer.__class__.__name__)
            inbound = [spiking_layers[inb.name] for inb in
                       layer.inbound_nodes[0].inbound_layers]
            if len(inbound) == 1:
                inbound = inbound[0]
            layer_kwargs = layer.get_config()
            layer_kwargs['config'] = self.config

            # If a preceding layer uses binary activations, add activation kwarg
            # to MaxPool layer because then we can use a cheaper operation.
            if hasattr(layer, 'activation') and 'binary' in \
                    layer.activation.__name__:
                binary_activation = layer.activation.__name__
            if 'MaxPool' in layer.name and binary_activation is not None:
                layer_kwargs['activation'] = binary_activation
                binary_activation = None

            # Create spiking layer and add to list.
            spiking_layers[layer.name] = spike_layer(**layer_kwargs)(inbound)

        print("Compiling spiking model...\n")
        self.snn = keras.models.Model(
            input_images, spiking_layers[parsed_model.layers[-1].name])
        self.snn.compile('sgd', 'categorical_crossentropy', ['accuracy'])
        self.snn.set_weights(parsed_model.get_weights())
        for layer in self.snn.layers:
            if hasattr(layer, 'b'):
                # Adjust biases to time resolution of simulator.
                layer.b.set_value(layer.b.get_value() *
                                  self.config['simulation']['dt'])
                if bias_relaxation:  # Experimental
                    layer.b0.set_value(layer.b.get_value())

        if self.fanin is None:
            from snntoolbox.core.util import get_ann_ops
            num_neurons, num_neurons_with_bias, fanin = self.set_connectivity()
            self.operations_ann = get_ann_ops(num_neurons,
                                              num_neurons_with_bias, fanin)
            print("Number of operations of ANN: {}\n".format(
                self.operations_ann))

    def run(self, x_test=None, y_test=None, dataflow=None, **kwargs):
        """
        Simulate a spiking network with non-leaky integrate-and-fire units,
        using a timestepped approach.

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
            Loads images from disk and processes them on the fly.

        kwargs: Optional[dict]
            - path: Optional[str]
                Where to store the output plots. If no path given, this value is
                taken from the settings dictionary.

        Returns
        -------

        top1acc_total: float
            Number of correctly classified samples divided by total number of
            test samples.
        """

        from snntoolbox.core.util import get_activations_batch, get_layer_ops
        from snntoolbox.core.util import echo
        from snntoolbox.io_utils.plotting import output_graphs
        from snntoolbox.io_utils.plotting import plot_confusion_matrix
        from snntoolbox.io_utils.plotting import plot_error_vs_time
        from snntoolbox.io_utils.plotting import plot_input_image
        from snntoolbox.io_utils.plotting import plot_ops_vs_time
        from snntoolbox.io_utils.AedatTools.DVSIterator import DVSIterator
        from snntoolbox.target_simulators.common import get_samples_from_list

        log_dir = kwargs[str('path')] if 'path' in kwargs \
            else self.config['paths']['log_dir_of_current_run']

        # Load neuron layers and connections if conversion was done during a
        # previous session.
        if self.snn is None:
            print("Restoring spiking network...\n")
            path_wd = self.config['paths']['path_wd']
            self.load(path_wd, self.config['paths']['filename_snn'])
            self.parsed_model = keras.models.load_model(os.path.join(
                path_wd, self.config['paths']['filename_parsed_model']+'.h5'))

        # Extract certain samples from test set, if user specified such a list.
        x_test, y_test = get_samples_from_list(x_test, y_test, dataflow,
                                               self.config)

        # Divide the test set into batches and run all samples in a batch in
        # parallel.
        batch_size = self.config.getint('simulation', 'batch_size')
        dataset_format = self.config['input']['dataset_format']
        num_batches = int(1e9) if dataset_format == 'aedat' else \
            int(np.floor(self.config.getint('simulation', 'num_to_test') /
                         batch_size))

        top5score_moving = 0
        score1_ann = 0
        score5_ann = 0
        truth_d = []  # Filled up with correct classes of all test samples.
        guesses_d = []  # Filled up with guessed classes of all test samples.
        guesses_b = np.zeros(batch_size)  # Guesses of one batch.
        x_b_xaddr = x_b_yaddr = x_b_ts = None
        x_b_l = y_b = None
        dvs_gen = None
        if dataset_format == 'aedat':
            dvs_gen = DVSIterator(
                self.config['paths']['dataset_path'], batch_size,
                eval(self.config['input']['label_dict']),
                eval(self.config['input']['subsample_facs']),
                self.config.getint('input', 'num_dvs_events_per_sample'))

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
        plot_keys = get_plot_keys(self.config)
        log_keys = get_log_keys(self.config)

        for batch_idx in range(num_batches):
            # Get a batch of samples
            if dataflow is not None and len(eval(
                    self.config['simulation']['sample_idxs_to_test'])) == 0:
                x_b_l, y_b = dataflow.next()
            elif not dataset_format == 'aedat':
                batch_idxs = range(batch_size * batch_idx,
                                   batch_size * (batch_idx + 1))
                x_b_l = x_test[batch_idxs, :]
                y_b = y_test[batch_idxs, :]
            if dataset_format == 'aedat':
                try:
                    x_b_xaddr, x_b_yaddr, x_b_ts, y_b = dvs_gen.__next__()
                except StopIteration:
                    break
                # Try to load frames here so we can plot input image later
                # TODO: Remove path hack!
                if any({'activations', 'correlation',
                        'hist_spikerates_activations'} & plot_keys) or \
                        'activations_n_b_l' in log_keys:
                    if x_test is None:
                        from snntoolbox.io_utils.common import load_npz
                        dataset_path_npz = '/home/rbodo/.snntoolbox/Datasets/' \
                                           'roshambo/frames_background'
                        x_test = load_npz(dataset_path_npz, 'x_test.npz')
                    batch_idxs = range(batch_size * batch_idx,
                                       batch_size * (batch_idx + 1))
                    x_b_l = x_test[batch_idxs, :]
            truth_b = np.argmax(y_b, axis=1)

            dt = self.config.getfloat('simulation', 'dt')
            # Either use Poisson spiketrains as inputs to the SNN, or take the
            # original data.
            poisson_input = self.config.getboolean('input', 'poisson_input')
            if poisson_input:
                # This factor determines the probability threshold for cells in
                # the input layer to fire a spike. Increasing ``input_rate``
                # increases the firing rate of the input and subsequent layers.
                self.rescale_fac = np.max(x_b_l) * 1000 / (
                   self.config.getint('input', 'input_rate') * dt)
            elif dataset_format == 'aedat':
                pass
            else:
                # Simply use the analog values of the original data as input.
                input_b_l = x_b_l * dt

            # Reset network variables.
            self.reset(batch_idx)

            # Allocate variables to monitor during simulation
            output_b_l = np.zeros((batch_size, self.num_classes), 'int32')

            input_spikecount = 0
            sim_step_int = 0
            duration = self.config.getint('simulation', 'duration')
            num_poisson_events_per_sample = \
                self.config.getint('input', 'num_poisson_events_per_sample')
            print("Starting new simulation...\n")
            # Loop through simulation time.
            for sim_step in range(1, duration + 1):
                sim_step *= dt
                # Generate input, in case it changes with each simulation step:
                if poisson_input:
                    if input_spikecount < num_poisson_events_per_sample \
                            or num_poisson_events_per_sample < 0:
                        spike_snapshot = np.random.random_sample(x_b_l.shape) \
                                         * self.rescale_fac
                        input_b_l = (spike_snapshot <= np.abs(x_b_l)).astype(
                            'float32')
                        input_spikecount += \
                            np.count_nonzero(input_b_l) / batch_size
                        # For BinaryNets, with input that is not normalized and
                        # not all positive, we stimulate with spikes of the same
                        # size as the maximum activation, and the same sign as
                        # the corresponding activation. Is there a better
                        # solution?
                        input_b_l *= np.max(x_b_l) * np.sign(x_b_l)
                    else:
                        input_b_l = np.zeros(x_b_l.shape)
                elif dataset_format == 'aedat':
                    input_b_l = np.zeros(self.snn.layers[0].batch_input_shape,
                                         'float32')
                    for sample_idx in range(batch_size):
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
                            if ts - first_ts_of_frame > self.config.getint(
                                    'input', 'eventframe_width'):
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
                if 'input_b_l_t' in log_keys:
                    self.input_b_l_t[Ellipsis, sim_step_int] = input_b_l
                if self.operations_b_t is not None:
                    if poisson_input or dataset_format == 'aedat':
                        input_ops = get_layer_ops(input_b_l, self.fanout[0])
                    else:
                        input_ops = np.ones(batch_size) * self.num_neurons[1]
                        if sim_step_int == 0:
                            input_ops *= 2 * self.fanin[1]  # MACs for convol.
                    self.operations_b_t[:, sim_step_int] += input_ops
                top1err = np.around(np.mean(self.top1err_b_t[:, sim_step_int]),
                                    4)
                sim_step_int += 1
                if self.config['output']['verbose'] > 0 and sim_step % 1 == 0:
                    echo('{:.2%}_'.format(1-top1err))

            num_samples_seen = (batch_idx + 1) * batch_size
            truth_d += list(truth_b)
            guesses_d += list(guesses_b)
            top1acc_moving = np.mean(np.array(truth_d) == np.array(guesses_d))
            top5score_moving += sum(in_top_k(output_b_l, truth_b, self.top_k))
            top5acc_moving = top5score_moving / num_samples_seen
            if self.config['output']['verbose'] > 0:
                print('\n')
                print("Batch {} of {} completed ({:.1%})".format(
                    batch_idx + 1, num_batches, (batch_idx + 1) / num_batches))
                print("Moving accuracy of SNN (top-1, top-5): {:.2%}, {:.2%}."
                      "".format(top1acc_moving, top5acc_moving))
            with open(path_acc, str('a')) as f_acc:
                f_acc.write(str("{} {:.2%} {:.2%}\n".format(
                    num_samples_seen, top1acc_moving, top5acc_moving)))

            # Evaluate ANN
            score = self.parsed_model.test_on_batch(x_b_l, y_b)
            score1_ann += score[1] * batch_size
            score5_ann += score[2] * batch_size
            self.top1err_ann = 1 - score1_ann / num_samples_seen
            self.top5err_ann = 1 - score5_ann / num_samples_seen
            print("Moving accuracy of ANN (top-1, top-5): {:.2%}, {:.2%}."
                  "\n".format(1 - self.top1err_ann, 1 - self.top5err_ann))

            if 'input_image' in plot_keys and x_b_l is not None:
                plot_input_image(x_b_l[0], int(truth_b[0]), log_dir)
            if 'error_t' in plot_keys:
                plot_error_vs_time(self.top1err_b_t, self.top5err_b_t,
                                   1 - self.top1err_ann, 1 - self.top5err_ann,
                                   log_dir)
            if 'confusion_matrix' in plot_keys:
                plot_confusion_matrix(truth_d, guesses_d, log_dir,
                                      list(np.arange(self.num_classes)))
            # Cumulate operation count over time and scale to MOps.
            if self.operations_b_t is not None:
                np.cumsum(np.divide(self.operations_b_t, 1e6), 1,
                          out=self.operations_b_t)
            if 'operations' in plot_keys:
                plot_ops_vs_time(self.operations_b_t, duration, dt, log_dir)
            if any({'activations', 'correlation', 'hist_spikerates_activations'}
                   & plot_keys) or 'activations_n_b_l' in log_keys:
                print("Calculating activations...\n")
                self.activations_n_b_l = get_activations_batch(
                    self.parsed_model, x_b_l)
            log_vars = {key: getattr(self, key) for key in log_keys}
            log_vars['top1err_b_t'] = self.top1err_b_t
            log_vars['top5err_b_t'] = self.top5err_b_t
            log_vars['top1err_ann'] = self.top1err_ann
            log_vars['top5err_ann'] = self.top5err_ann
            log_vars['operations_ann'] = self.operations_ann / 1e6
            np.savez_compressed(os.path.join(path_log_vars, str(batch_idx)),
                                **log_vars)
            plot_vars = {}
            if any({'activations', 'correlation',
                    'hist_spikerates_activations'} & plot_keys):
                plot_vars['activations_n_b_l'] = self.activations_n_b_l
            if any({'spiketrains', 'spikerates', 'correlation', 'spikecounts',
                    'hist_spikerates_activations'} & plot_keys):
                plot_vars['spiketrains_n_b_l_t'] = self.spiketrains_n_b_l_t
            output_graphs(plot_vars, self.config, log_dir, 0)
        # Compute average accuracy, taking into account number of samples per
        # class
        count = np.zeros(self.num_classes)
        match = np.zeros(self.num_classes)
        for gt, p in zip(truth_d, guesses_d):
            count[gt] += 1
            if gt == p:
                match[gt] += 1
        avg_acc = np.mean(np.true_divide(match, count))
        top1acc_total = np.mean(np.array(truth_d) == np.array(guesses_d))
        if 'confusion_matrix' in plot_keys:
            plot_confusion_matrix(truth_d, guesses_d, log_dir,
                                  list(np.arange(self.num_classes)))
        print("Simulation finished.\n\n")
        print("Total accuracy: {:.2%} on {} test samples.\n\n".format(
            top1acc_total, len(guesses_d)))
        print("Accuracy averaged over classes: {:.2%}".format(avg_acc))

        return top1acc_total

    def save(self, path, filename):
        """Write model architecture and parameters to disk.

        Parameters
        ----------

        path: str
            Path to directory where to save model to.

        filename: str
            Name of file to write model to.
        """

        filepath = os.path.join(path, filename + '.h5')

        print("Saving model to {}...\n".format(filepath))
        self.snn.save(filepath, self.config.getboolean('output', 'overwrite'))

    def load(self, path, filename):
        """Load model architecture and parameters from disk.

        Sets the ``snn`` attribute of this class.

        Parameters
        ----------

        path: str
            Path to directory where to load model from.

        filename: str
            Name of file to load model from.
        """

        from snntoolbox.core.inisim import custom_layers

        filepath = os.path.join(path, filename + '.h5')

        # TODO: Loading does not work anymore because the configparser object
        # needed by the custom layers is not stored when saving the model.
        # Could be implemented by overriding Keras' save / load methods, but
        # since converting even large Keras models from scratch is so fast,
        # there's really no need.
        self.snn = keras.models.load_model(filepath, custom_layers)

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

        num_timesteps = int(self.config.getint('simulation', 'duration') /
                            self.config.getfloat('simulation', 'dt'))
        batch_size = self.config.getint('simulation', 'batch_size')
        plot_keys = get_plot_keys(self.config)
        log_keys = get_log_keys(self.config)

        if 'input_b_l_t' in log_keys:
            self.input_b_l_t = np.empty(
                list(self.snn.input_shape) + [num_timesteps], np.bool)

        if any({'spiketrains', 'spikerates', 'correlation', 'spikecounts',
                'hist_spikerates_activations'} & plot_keys) \
                or 'spiketrains_n_b_l_t' in log_keys:
            self.spiketrains_n_b_l_t = []
            for layer in self.snn.layers:
                if not hasattr(layer, 'spiketrain'):
                    continue
                shape = list(layer.output_shape) + [num_timesteps]
                self.spiketrains_n_b_l_t.append((np.zeros(shape, 'float32'),
                                                 layer.name))

        if 'operations' in plot_keys or 'operations_b_t' in log_keys:
            self.operations_b_t = np.zeros((batch_size, num_timesteps))

        if 'mem_n_b_l_t' in log_keys or 'mem' in plot_keys:
            self.mem_n_b_l_t = []
            for layer in self.snn.layers:
                if not hasattr(layer, 'mem'):
                    continue
                shape = list(layer.output_shape) + [num_timesteps]
                self.mem_n_b_l_t.append((np.zeros(shape, 'float32'),
                                         layer.name))

        self.top1err_b_t = np.empty((batch_size, num_timesteps), np.bool)
        self.top5err_b_t = np.empty((batch_size, num_timesteps), np.bool)

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
