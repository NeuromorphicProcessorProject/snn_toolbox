# coding=utf-8

"""Common functions for spiking simulators."""

import os
import numpy as np
from abc import abstractmethod
from snntoolbox.core.util import is_spiking, get_plot_keys, get_log_keys, echo
from snntoolbox.core.util import get_layer_ops


class AbstractSNN:
    """
    The compiled spiking neural network, using layers derived from
    Keras base classes.

    Aims at simulating the network on a self-implemented Integrate-and-Fire
    simulator using a timestepped approach.

    Attributes
    ----------

    sim: Simulator
        Module containing utility functions of spiking simulator. Result of
        calling ``snntoolbox.config.initialize_simulator()``. For instance, if
        using Brian simulator, this initialization would be equivalent to
        ``import pyNN.brian as sim``.

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
        from snntoolbox.config import initialize_simulator

        self.config = config
        self.queue = queue
        self.parsed_model = None
        self.is_built = False
        self._batch_size = None  # Store original batch_size here.
        self.batch_size = self.adjust_batchsize()

        # Logging variables
        self.spiketrains_n_b_l_t = self.activations_n_b_l = None
        self.input_b_l_t = self.mem_n_b_l_t = None
        self.top1err_b_t = self.top5err_b_t = None
        self.operations_b_t = self.operations_ann = None
        self.top1err_ann = self.top5err_ann = None
        self.num_neurons = self.num_neurons_with_bias = None
        self.fanin = self.fanout = None
        self._dt = self.config.getfloat('simulation', 'dt')
        self._duration = self.config.getint('simulation', 'duration')
        self._num_timesteps = int(self._duration / self._dt)
        # ``rescale_fac`` globally scales spike probability when using Poisson
        # input.
        self.rescale_fac = 1000 / (self.config.getint('input', 'input_rate') *
                                   self._dt)
        self.num_classes = None
        self.top_k = None

        self.sim = initialize_simulator(config['simulation']['simulator'],
                                        dt=config.getfloat('simulation', 'dt'))

        self._dataset_format = self.config['input']['dataset_format']
        self._poisson_input = self.config.getboolean('input', 'poisson_input')
        self._num_poisson_events_per_sample = \
            self.config.getint('input', 'num_poisson_events_per_sample')
        self._input_spikecount = 0

        self._plot_keys = get_plot_keys(self.config)
        self._log_keys = get_log_keys(self.config)
        self._mem_container_counter = None
        self._spiketrains_container_counter = None

    @property
    @abstractmethod
    def is_parallelizable(self):
        pass

    @abstractmethod
    def add_input_layer(self, input_shape):
        pass

    @abstractmethod
    def add_layer(self, layer):
        pass

    @abstractmethod
    def build_dense(self, layer):
        pass

    @abstractmethod
    def build_convolution(self, layer):
        pass

    @abstractmethod
    def build_pooling(self, layer):
        pass

    def build_flatten(self, layer):
        pass

    @abstractmethod
    def compile(self):
        pass

    @abstractmethod
    def simulate(self, **kwargs):
        """

        Returns
        -------

        output_b_l_t: ndarray
            Array of shape (batch_size, num_classes, num_timesteps), containing
            the number of output spikes of the neurons in the final layer, for
            each sample and for each time step during the simulation.
        """

        pass

    @abstractmethod
    def reset(self, sample_idx):
        """Reset network variables."""

        pass

    @abstractmethod
    def end_sim(self):
        """Clean up after simulation."""
        pass

    @abstractmethod
    def save(self, path, filename):
        """Write model architecture and parameters to disk.

        Parameters
        ----------

        path: string
            Path to directory where to save model.

        filename: string
            Name of file to write model to.
        """

        pass

    @abstractmethod
    def load(self, path, filename):
        """Load model architecture and parameters from disk.

        Parameters
        ----------

        path: str
            Path to directory where to load model from.

        filename: str
            Name of file to load model from.
        """

        pass

    def build(self, parsed_model):
        """
        Compile a spiking neural network to prepare for simulation.

        Written in pyNN (http://neuralensemble.org/docs/PyNN/).
        pyNN is a simulator-independent language for building neural
        network
        models. It allows running the converted net in a Spiking
        Simulator like
        Brian, NEURON, or NEST.

        During compilation, two lists are created and stored to disk:
        ``layers`` and ``connections``. Each entry in ``layers``
        represents a
        population of neurons, given by a pyNN ``Population`` object. The
        neurons in these layers are connected by pyNN ``Projection`` s,
        stored
        in ``connections`` list.

        This compilation method performs the connection process between
        layers.
        This means, if the session was started with a call to
        ``sim.setup()``,
        the converted network can be tested right away, using the simulator
        ``sim``.

        However, when starting a new session (calling ``sim.setup()`` after
        conversion), the ``layers`` have to be reloaded from disk using
        ``load_assembly``, and the connections reestablished manually.
        This is
        implemented in ``run`` method, go there for details.
        See ``snntoolbox.core.pipeline.test_full`` about how to simulate
        after
        converting.

        Parameters
        ----------

        parsed_model: Keras model
         Parsed input model.
        """

        from snntoolbox.core.util import get_type

        print("Building spiking model...")

        self.parsed_model = parsed_model
        self.num_classes = int(self.parsed_model.layers[-1].output_shape[-1])
        self.top_k = min(self.num_classes, 5)

        # Get batch input shape
        batch_shape = list(parsed_model.layers[0].batch_input_shape)
        batch_shape[0] = self.batch_size

        self.add_input_layer(batch_shape)

        # Iterate over layers to create spiking neurons and connections.
        for layer in parsed_model.layers[1:]:
            print("Building layer: {}".format(layer.name))
            self.add_layer(layer)
            layer_type = get_type(layer)
            if layer_type == 'Dense':
                self.build_dense(layer)
            elif layer_type == 'Conv2D':
                self.build_convolution(layer)
            elif layer_type in {'MaxPooling2D', 'AveragePooling2D'}:
                self.build_pooling(layer)
            elif layer_type == 'Flatten':
                self.build_flatten(layer)

        print("Compiling spiking model...\n")
        self.compile()

        # Compute number of operations of ANN.
        if self.fanin is None:
            from snntoolbox.core.util import get_ann_ops
            num_neurons, num_neurons_with_bias, fanin = self.set_connectivity()
            self.operations_ann = get_ann_ops(num_neurons,
                                              num_neurons_with_bias, fanin)
            print("Number of operations of ANN: {}\n".format(
                self.operations_ann))

        self.is_built = True

    def run(self, x_test=None, y_test=None, dataflow=None, **kwargs):
        """
        Simulate a spiking network.

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

        from snntoolbox.core.util import get_activations_batch, in_top_k
        import snntoolbox.io_utils.plotting as snn_plt

        # Get directory where logging quantities will be stored.
        log_dir = kwargs[str('path')] if 'path' in kwargs \
            else self.config['paths']['log_dir_of_current_run']

        # Load neuron layers and connections if conversion was done during a
        # previous session.
        if not self.is_built:
            self.restore_snn()

        # Extract certain samples from test set, if user specified such a list.
        x_test, y_test = get_samples_from_list(x_test, y_test, dataflow,
                                               self.config)

        # Divide the test set into batches and run all samples in a batch in
        # parallel.
        dataset_format = self.config['input']['dataset_format']
        num_batches = int(1e9) if dataset_format == 'aedat' else \
            int(np.floor(self.config.getint('simulation', 'num_to_test') /
                         self.batch_size))

        # Initialize intermediate variables for computing statistics.
        top5score_moving = 0
        score1_ann = 0
        score5_ann = 0
        truth_d = []  # Filled up with correct classes of all test samples.
        guesses_d = []  # Filled up with guessed classes of all test samples.

        # Prepare files for storage of logging quantities.
        path_log_vars = os.path.join(log_dir, 'log_vars')
        if not os.path.isdir(path_log_vars):
            os.makedirs(path_log_vars)
        path_acc = os.path.join(log_dir, 'accuracy.txt')
        if os.path.isfile(path_acc):
            os.remove(path_acc)

        self.init_log_vars()

        self.init_cells()

        # This dict will be used to pass a batch of data to the simulator.
        data_batch_kwargs = {}

        # If DVS events are used as input, instantiate a DVSIterator.
        if dataset_format == 'aedat':
            from snntoolbox.io_utils.AedatTools.DVSIterator import DVSIterator
            batch_shape = self.parsed_model.get_batch_input_shape()
            batch_shape[0] = self.batch_size
            dvs_gen = DVSIterator(
                self.config['paths']['dataset_path'], batch_shape,
                self.config.getint('input', 'eventframe_width'),
                self.config.getint('input', 'num_dvs_events_per_sample'),
                batch_shape[2:], eval(self.config['input']['target_size']),
                eval(self.config['input']['label_dict']))
            data_batch_kwargs['dvs_gen'] = dvs_gen

        # Simulate the SNN on a batch of samples in parallel.
        for batch_idx in range(num_batches):

            # Get a batch of samples
            if x_test is not None:
                batch_idxs = range(self.batch_size * batch_idx,
                                   self.batch_size * (batch_idx + 1))
                x_b_l = x_test[batch_idxs, :]
                if y_test is not None:
                    y_b = y_test[batch_idxs, :]
            elif dataflow is not None:
                x_b_l, y_b = dataflow.next()
            elif dataset_format == 'aedat':
                try:
                    data_batch_kwargs['dvs_gen'].next_sequence_batch()
                    y_b = data_batch_kwargs['dvs_gen'].y_b
                except StopIteration:
                    break

                # Generate frames so we can compare with ANN.
                if any({'activations', 'correlation', 'input_image',
                        'hist_spikerates_activations'} & self._plot_keys) or \
                        'activations_n_b_l' in self._log_keys:
                    x_b_l = data_batch_kwargs['dvs_gen'].get_frames_batch()

            truth_b = np.argmax(y_b, axis=1)
            data_batch_kwargs['truth_b'] = truth_b
            data_batch_kwargs['x_b_l'] = x_b_l

            # Main step: Run the network on a batch of samples for the duration
            # of the simulation.
            print("Starting new simulation...\n")
            output_b_l_t = self.simulate(**data_batch_kwargs)

            # Get classification result by comparing the guessed class (i.e. the
            # index of the neuron in the last layer which spiked most) to the
            # ground truth.
            guesses_b_t = np.argmax(output_b_l_t, axis=1)
            # Find sample indices for which there was no output spike yet.
            undecided_b_t = np.nonzero(np.sum(output_b_l_t, 1) == 0)
            # Assign negative value such that undecided samples count as
            # wrongly classified.
            guesses_b_t[undecided_b_t] = -1

            # Get classification error of current batch, for each time step.
            self.top1err_b_t = guesses_b_t != np.broadcast_to(
                np.expand_dims(truth_b, 1), guesses_b_t.shape)
            for t in range(self._num_timesteps):
                self.top5err_b_t[:, t] = ~in_top_k(output_b_l_t[:, :, t],
                                                   truth_b, self.top_k)

            # Add results of current batch to previous results.
            truth_d += list(truth_b)
            guesses_d += list(guesses_b_t[:, -1])

            # Print current accuracy.
            num_samples_seen = (batch_idx + 1) * self.batch_size
            top1acc_moving = np.mean(np.array(truth_d) == np.array(guesses_d))
            top5score_moving += sum(in_top_k(output_b_l_t[:, :, -1], truth_b,
                                             self.top_k))
            top5acc_moving = top5score_moving / num_samples_seen
            if self.config.getint('output', 'verbose') > 0:
                print("\nBatch {} of {} completed ({:.1%})".format(
                    batch_idx + 1, num_batches, (batch_idx + 1) / num_batches))
                print("Moving accuracy of SNN (top-1, top-5): {:.2%}, {:.2%}."
                      "".format(top1acc_moving, top5acc_moving))
            with open(path_acc, str('a')) as f_acc:
                f_acc.write(str("{} {:.2%} {:.2%}\n".format(
                    num_samples_seen, top1acc_moving, top5acc_moving)))

            # Evaluate ANN on the same batch as SNN for a direct comparison.
            score = self.parsed_model.test_on_batch(x_b_l, y_b)
            score1_ann += score[1] * self.batch_size
            score5_ann += score[2] * self.batch_size
            self.top1err_ann = 1 - score1_ann / num_samples_seen
            self.top5err_ann = 1 - score5_ann / num_samples_seen
            print("Moving accuracy of ANN (top-1, top-5): {:.2%}, {:.2%}."
                  "\n".format(1 - self.top1err_ann, 1 - self.top5err_ann))

            # Plot input image.
            if 'input_image' in self._plot_keys:
                snn_plt.plot_input_image(x_b_l[0], int(truth_b[0]), log_dir)

            # Plot error vs time.
            if 'error_t' in self._plot_keys:
                snn_plt.plot_error_vs_time(
                    self.top1err_b_t, self.top5err_b_t, self._duration,
                    self._dt, 1 - self.top1err_ann, 1 - self.top5err_ann,
                    log_dir)

            # Plot confusion matrix.
            if 'confusion_matrix' in self._plot_keys:
                snn_plt.plot_confusion_matrix(truth_d, guesses_d, log_dir,
                                              list(np.arange(self.num_classes)))

            # Cumulate operation count over time and scale to MOps.
            if self.operations_b_t is not None:
                np.cumsum(np.divide(self.operations_b_t, 1e6), 1,
                          out=self.operations_b_t)

            # Plot operations vs time.
            if 'operations' in self._plot_keys:
                snn_plt.plot_ops_vs_time(self.operations_b_t, self._duration,
                                         self._dt, log_dir)

            # Calculate ANN activations for plots.
            if any({'activations', 'correlation', 'hist_spikerates_activations'}
                   & self._plot_keys) or 'activations_n_b_l' in self._log_keys:
                print("Calculating activations...\n")
                self.activations_n_b_l = get_activations_batch(
                    self.parsed_model, x_b_l)

            # Save log variables to disk.
            log_vars = {key: getattr(self, key) for key in self._log_keys}
            log_vars['top1err_b_t'] = self.top1err_b_t
            log_vars['top5err_b_t'] = self.top5err_b_t
            log_vars['top1err_ann'] = self.top1err_ann
            log_vars['top5err_ann'] = self.top5err_ann
            log_vars['operations_ann'] = self.operations_ann / 1e6
            np.savez_compressed(os.path.join(path_log_vars, str(batch_idx)),
                                **log_vars)

            # More plotting.
            plot_vars = {}
            if any({'activations', 'correlation',
                    'hist_spikerates_activations'} & self._plot_keys):
                plot_vars['activations_n_b_l'] = self.activations_n_b_l
            if any({'spiketrains', 'spikerates', 'correlation', 'spikecounts',
                    'hist_spikerates_activations'} & self._plot_keys):
                plot_vars['spiketrains_n_b_l_t'] = self.spiketrains_n_b_l_t
            snn_plt.output_graphs(plot_vars, self.config, log_dir, 0)

            # Reset network variables.
            self.reset(batch_idx)

        # Plot confusion matrix for whole data set.
        if 'confusion_matrix' in self._plot_keys:
            snn_plt.plot_confusion_matrix(truth_d, guesses_d, log_dir,
                                          list(np.arange(self.num_classes)))

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

        # Print final result.
        print("Simulation finished.\n\n")
        ss = '' if self.config.getint('simulation', 'num_to_test') == 1 else 's'
        print("Total accuracy: {:.2%} on {} test sample{}.\n\n".format(
            top1acc_total, len(guesses_d), ss))
        print("Accuracy averaged over classes: {:.2%}".format(avg_acc))

        # If batch_size was modified, change back to original value now.
        if self.batch_size != self._batch_size:
            self.config.set('simulation', 'batch_size', str(self._batch_size))

        return top1acc_total

    def init_cells(self):
        """
        Set cellparameters of neurons in each layer and initialize membrane
        potential.
        """

        pass

    def adjust_batchsize(self):
        self._batch_size = self.config.getint('simulation', 'batch_size')
        if self._batch_size > 1 and not self.is_parallelizable:
            self.config.set('simulation', 'batch_size', '1')
            print("Temporarily setting batch_size to 1 because simulator does "
                  "not support parallel testing of multiple samples.")
            return 1
        return self._batch_size

    def restore_snn(self):
        import keras
        print("Restoring spiking network...\n")
        self.load(self.config['paths']['path_wd'],
                  self.config['paths']['filename_snn'])
        self.parsed_model = keras.models.load_model(os.path.join(
            self.config['paths']['path_wd'],
            self.config['paths']['filename_parsed_model'] + '.h5'))

    def init_log_vars(self):
        """Initialize debug variables."""

        if 'input_b_l_t' in self._log_keys:
            self.input_b_l_t = np.empty(list(self.parsed_model.input_shape) +
                                        [self._num_timesteps], np.bool)

        if any({'spiketrains', 'spikerates', 'correlation', 'spikecounts',
                'hist_spikerates_activations'} & self._plot_keys) \
                or 'spiketrains_n_b_l_t' in self._log_keys:
            self.spiketrains_n_b_l_t = []
            for layer in self.parsed_model.layers:
                if not is_spiking(layer):
                    continue
                shape = list(layer.output_shape) + [self._num_timesteps]
                self.spiketrains_n_b_l_t.append((np.zeros(shape, 'float32'),
                                                 layer.name))

        if 'operations' in self._plot_keys or \
                'operations_b_t' in self._log_keys:
            self.operations_b_t = np.zeros((self.batch_size,
                                            self._num_timesteps))

        if 'mem_n_b_l_t' in self._log_keys or 'mem' in self._plot_keys:
            self.mem_n_b_l_t = []
            for layer in self.parsed_model.layers:
                if not is_spiking(layer):
                    continue
                shape = list(layer.output_shape) + [self._num_timesteps]
                self.mem_n_b_l_t.append((np.zeros(shape, 'float32'),
                                         layer.name))

        self.top1err_b_t = np.empty((self.batch_size, self._num_timesteps),
                                    np.bool)
        self.top5err_b_t = np.empty((self.batch_size, self._num_timesteps),
                                    np.bool)

    def set_connectivity(self):
        """
        Set connectivity statistics needed to compute the number of operations
        in the network, e.g. fanin, fanout, number_of_neurons,
        number_of_neurons_with_bias.
        """

        from snntoolbox.core.util import get_fanin, get_fanout, get_type

        first_hidden_layer = self.parsed_model.layers[1]
        self.fanin = [0]
        if 'Dense' in get_type(first_hidden_layer):
            self.fanout = [first_hidden_layer.units]
        else:
            self.fanout = [int(np.multiply(np.prod(
                first_hidden_layer.kernel_size), first_hidden_layer.filters))]
        self.num_neurons = [np.product(self.parsed_model.input_shape[1:])]
        self.num_neurons_with_bias = [0]

        for layer in self.parsed_model.layers:
            if is_spiking(layer):
                self.fanin.append(get_fanin(layer))
                self.fanout.append(get_fanout(layer))
                self.num_neurons.append(np.prod(layer.output_shape[1:]))
                if hasattr(layer, 'b') and any(layer.b.get_value()):
                    print("Detected layer with biases: {}".format(layer.name))
                    self.num_neurons_with_bias.append(self.num_neurons[-1])
                else:
                    self.num_neurons_with_bias.append(0)

        return self.num_neurons, self.num_neurons_with_bias, self.fanin

    def get_recorded_vars(self, layers):
        """Retrieve neuron variables recorded during simulation."""

        self.set_spiketrain_stats_input()

        self.reset_container_counters()

        for i in range(len(layers)):
            kwargs = {'layer': layers[i], 'monitor_index': i}
            spiketrains_b_l_t = self.get_spiketrains(**kwargs)
            if spiketrains_b_l_t is not None:
                self.set_spiketrain_stats(spiketrains_b_l_t)

            mem = self.get_vmem(**kwargs)
            if mem is not None:
                self.set_mem_stats(mem)

        # For each time step, get number of spikes of all neurons in the output
        # layer.
        shape = (self.batch_size, self.num_classes, self._num_timesteps)
        output_b_l_t = np.zeros(shape, 'int32')
        kwargs = {'layer': layers[-1], 'monitor_index': -1}
        # Need to reduce counter here to be able to access the last monitor.
        # TODO: Remove this ugly hack!
        self._spiketrains_container_counter -= 1
        spiketrains_b_l_t = self.get_spiketrains(**kwargs)
        self._spiketrains_container_counter += 1
        for b in range(shape[0]):
            for l in range(shape[1]):
                for t in range(shape[2]):
                    output_b_l_t[b, l, t] = np.count_nonzero(
                        spiketrains_b_l_t[b, l, :t])
        return output_b_l_t

    def reset_container_counters(self):
        self._mem_container_counter = 0
        self._spiketrains_container_counter = 0

    def get_spiketrains(self, **kwargs):
        """Get spike trains of a layer.

        Returns
        -------

        spiketrains_b_l_t : ndarray
            Spike trains.
        """

        pass

    @abstractmethod
    def get_spiketrains_input(self):
        """Get spike trains of input layer.

        Returns
        -------

        spiketrains_b_l_t : ndarray
            Spike trains of input.
        """

        pass

    def get_vmem(self, layer, i):
        pass

    def set_mem_stats(self, mem):

        from snntoolbox.io_utils.plotting import plot_potential

        # Reshape flat array to original layer shape.
        i = self._mem_container_counter
        self.mem_n_b_l_t[i] = (np.reshape(mem, self.mem_n_b_l_t[i][0].shape),
                               self.mem_n_b_l_t[i][1])

        # Plot membrane potentials of layer.
        times = self._dt * np.arange(self._num_timesteps)
        show_legend = True if i >= len(self.mem_n_b_l_t) - 2 else False
        plot_potential(times, self.mem_n_b_l_t[i], self.config, show_legend,
                       self.config['paths']['log_dir_of_current_run'])

        self._mem_container_counter += 1

    def set_spiketrain_stats_input(self):

        if self._poisson_input or self._dataset_format == 'aedat':
            spiketrains_b_l_t = self.get_spiketrains_input()
            if self.input_b_l_t is not None:
                self.input_b_l_t = spiketrains_b_l_t
            if self.operations_b_t is not None:
                for t in range(self._num_timesteps):
                    self.operations_b_t[:, t] += get_layer_ops(
                        spiketrains_b_l_t[Ellipsis, t], self.fanout[0])
        else:
            if self.input_b_l_t is not None:
                raise NotImplementedError
            if self.operations_b_t is not None:
                input_ops = np.ones(self.batch_size) * self.num_neurons[1]
                self.operations_b_t[:, 0] += input_ops * self.fanin[1] * 2
                for t in range(self._num_timesteps):
                    self.operations_b_t[:, t] += input_ops

    def set_spiketrain_stats(self, spiketrains_b_l_t):

        # Add spike trains to log variables.
        if self.spiketrains_n_b_l_t is not None:
            i = self._spiketrains_container_counter
            self.spiketrains_n_b_l_t[i] = (spiketrains_b_l_t,
                                           self.spiketrains_n_b_l_t[i][1])
            self._spiketrains_container_counter += 1

        # Use spike trains to compute the number of operations.
        if self.operations_b_t is not None:
            for t in range(self._num_timesteps):
                self.operations_b_t[:, t] += get_layer_ops(
                    spiketrains_b_l_t[Ellipsis, t], self.fanout[i + 1],
                    self.num_neurons_with_bias[i + 1])

    def reshape_flattened_spiketrains(self, spiketrains, shape):
        """
        Convert list of spike times into array where nonzero entries (indicating
        spike times) are properly spread out across array. Then reshape the flat
        array into original layer shape.
        """

        spiketrains_flat = np.zeros((np.prod(shape[:-1]), shape[-1]))
        for k, spiketrain in enumerate(spiketrains):
            for t in spiketrain:
                spiketrains_flat[k, int(t / self._dt)] = t

        spiketrains_b_l_t = np.reshape(spiketrains_flat, shape)

        return spiketrains_b_l_t


def get_samples_from_list(x_test, y_test, dataflow, config):
    """
    If user specified a list of samples to test with
    ``settings['sample_idxs_to_test']``, this function extract them from the
    test set.
    """

    batch_size = config.getint('simulation', 'batch_size')
    si = list(eval(config['simulation']['sample_idxs_to_test']))
    if not len(si) == 0:
        if dataflow is not None:
            batch_idx = 0
            x_test = []
            y_test = []
            target_idx = si.pop(0)
            while len(x_test) < config.getint('simulation', 'num_to_test'):
                x_b_l, y_b = dataflow.next()
                for i in range(batch_size):
                    if batch_idx * batch_size + i == target_idx:
                        x_test.append(x_b_l[i])
                        y_test.append(y_b[i])
                        if len(si) > 0:
                            target_idx = si.pop(0)
                batch_idx += 1
            x_test = np.array(x_test)
            y_test = np.array(y_test)
        elif x_test is not None:
            x_test = np.array([x_test[i] for i in si])
            y_test = np.array([y_test[i] for i in si])

    return x_test, y_test


def build_convolution(layer, delay):
    """Build convolution layer."""

    weights, biases = layer.get_weights()

    # Biases.
    i_offset = np.empty(np.prod(layer.output_shape[1:]))
    n = int(len(i_offset) / len(biases))
    for i in range(len(biases)):
        i_offset[i:(i + 1) * n] = biases[i]

    nx = layer.input_shape[3]  # Width of feature map
    ny = layer.input_shape[2]  # Height of feature map
    kx, ky = layer.kernel_size  # Width and height of kernel
    px = int((kx - 1) / 2)  # Zero-padding columns
    py = int((ky - 1) / 2)  # Zero-padding rows

    if layer.padding == 'valid':
        # In padding 'valid', the original sidelength is
        # reduced by one less than the kernel size.
        mx = nx - kx + 1  # Number of columns in output filters
        my = ny - ky + 1  # Number of rows in output filters
        x0 = px
        y0 = py
    elif layer.padding == 'same':
        mx = nx
        my = ny
        x0 = 0
        y0 = 0
    else:
        raise NotImplementedError("Border_mode {} not supported".format(
            layer.padding))

    connections = []

    # Loop over output filters 'fout'
    for fout in range(weights.shape[3]):
        for y in range(y0, ny - y0):
            for x in range(x0, nx - x0):
                target = x - x0 + (y - y0) * mx + fout * mx * my
                # Loop over input filters 'fin'
                for fin in range(weights.shape[2]):
                    for k in range(-py, py + 1):
                        if not 0 <= y + k < ny:
                            continue
                        source = x + (y + k) * nx + fin * nx * ny
                        for l in range(-px, px + 1):
                            if not 0 <= x + l < nx:
                                continue
                            connections.append((source + l, target,
                                                weights[py - k, px - l, fin,
                                                        fout], delay))
        echo('.')
    print('')

    return connections, i_offset


def build_pooling(layer, delay):
    """Build pooling layer."""

    if layer.__class__.__name__ == 'MaxPooling2D':
        import warnings

        warnings.warn("Layer type 'MaxPooling' not supported yet. " +
                      "Falling back on 'AveragePooling'.", RuntimeWarning)

    nx = layer.input_shape[3]  # Width of feature map
    ny = layer.input_shape[2]  # Hight of feature map
    dx = layer.pool_size[1]  # Width of pool
    dy = layer.pool_size[0]  # Hight of pool
    sx = layer.strides[1]
    sy = layer.strides[0]

    connections = []

    for fout in range(layer.input_shape[1]):  # Feature maps
        for y in range(0, ny - dy + 1, sy):
            for x in range(0, nx - dx + 1, sx):
                target = int(x / sx + y / sy * ((nx - dx) / sx + 1) +
                             fout * nx * ny / (dx * dy))
                for k in range(dy):
                    source = x + (y + k) * nx + fout * nx * ny
                    for l in range(dx):
                        connections.append((source + l, target, 1 / (dx * dy),
                                            delay))
        echo('.')
    print('')

    return connections
