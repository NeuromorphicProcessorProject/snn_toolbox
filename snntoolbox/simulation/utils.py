# coding=utf-8

"""Common functions for spiking simulators.

Most notably, this module defines the abstract base class `AbstractSNN` used to
create spiking neural networks. This class has to be inherited from when another
simulator is added to the toolbox (see :ref:`extending`).

@author: rbodo
"""

import os
from abc import abstractmethod

import numpy as np

from snntoolbox.bin.utils import get_log_keys, get_plot_keys
from snntoolbox.parsing.utils import get_type
from snntoolbox.utils.utils import echo


class AbstractSNN:
    """Abstract base class for creating spiking neural networks.

    This class provides the basic structure to compile and simulate a spiking
    neural network. It has to be instantiated as in
    `target_simulators.pyNN_target_sim` with concrete methods tailored to the
    target simulator.

    Core methods that usually do not have to be overwritten include:

    .. autosummary::
        :nosignatures:

        build
        run
        get_recorded_vars

    Relevant methods that in most cases will have to be overwritten include:

    .. autosummary::
        :nosignatures:

        add_input_layer
        add_layer
        build_dense
        build_convolution
        build_pooling
        build_flatten
        compile
        simulate

    Notes
    -----

    In the attribute definitions below, we use suffixes to denote variable
    shape. In `spiketrains_n_b_l_t` for instance, ``n`` represents a dimension
    across the network (i.e. indexing the layers); ``b`` is an index for a
    batch of samples; ``l`` stands for the layer dimensions; ``t``
    indicates the time axis.

    Attributes
    ----------

    config: configparser.ConfigParser
        Settings.
    queue: Queue.Queue
        Used to detect stop signals from user to abort simulation.
    parsed_model: keras.models.Model
        The parsed model.
    is_built: bool
        Whether or not the SNN has been built.
    batch_size: int
        The batch size for parallel testing of multiple samples.
    spiketrains_n_b_l_t: list[tuple[np.array, str]]
        Spike trains of a batch of samples of all neurons in the network over
        the whole simulation time. Each entry in ``spiketrains_batch`` contains
        a tuple ``(spiketimes, label)`` for each layer of the network.
        ``spiketimes`` is an array where the last index contains the spike times
        of the specific neuron, and the first indices run over the number of
        neurons in the layer: (batch_size, n_chnls, n_rows, n_cols, duration).
        ``label`` is a string specifying both the layer type and the index,
        e.g. ``'03Conv2D_32x64x64'``.
    activations_n_b_l: list[tuple[np.array, str]]
        Activations of the ANN.
    mem_n_b_l_t: list[tuple[np.array, str]]
        Membrane potentials of the SNN.
    input_b_l_t: ndarray
        Input to the SNN over time.
    top1err_b_t: ndarray
        Top-1 error of SNN over time. Shape: (`batch_size`, ``_num_timesteps``).
    top5err_b_t: ndarray
        Top-5 error of SNN over time. Shape: (`batch_size`, ``_num_timesteps``).
    operations_b_t: ndarray
        Number of operations of SNN over time. Shape:
        (`batch_size`, ``_num_timesteps``)
    operations_ann: float
        Number of operations of ANN.
    top1err_ann: float
        Top-1 error of ANN.
    top5err_ann: float
        Top-5 error of ANN.
    num_neurons: List[int]
        Number of neurons in the network (one entry per layer).
    num_neurons_with_bias:
        Number of neurons with bias in the network (one entry per layer).
    fanin: List[int]
        Number of synapses targeting a neuron (one entry per layer).
    fanout: List[int]
        Number of outgoing synapses (one entry per layer).
    rescale_fac: float
        Scales spike probability when using Poisson input.
    num_classes: int
        Number of classes of the data set.
    top_k: int
        By default, the toolbox records the top-1 and top-k classification
        errors.
    sim: Simulator
        Module containing utility functions of spiking simulator. Result of
        calling :py:func:`snntoolbox.bin.utils.initialize_simulator`. For
        instance, if using Brian simulator, this initialization would be
        equivalent to ``import pyNN.brian as sim``.
    """

    def __init__(self, config, queue=None):
        from snntoolbox.bin.utils import initialize_simulator

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
        self.num_neurons = self.num_neurons_with_bias = self.num_synapses = None
        self.fanin = self.fanout = None
        self._dt = self.config.getfloat('simulation', 'dt')
        self._duration = self.config.getint('simulation', 'duration')
        self._num_timesteps = int(self._duration / self._dt)
        self.rescale_fac = 1000 / (self.config.getint('input', 'input_rate') *
                                   self._dt)
        self.num_classes = None
        self.top_k = None

        self.sim = initialize_simulator(config)

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
        """
        Whether or not the simulator is able to test multiple samples in
        parallel.
        """

        pass

    @abstractmethod
    def add_input_layer(self, input_shape):
        """Add input layer.

        Parameters
        ----------

        input_shape: tuple
            Input shape to the network, including the batch size as first
            dimension.
        """

        pass

    @abstractmethod
    def add_layer(self, layer):
        """Do anything that concerns adding any layer independently of its type.

        Parameters
        ----------

        layer: keras.layers.Layer
            Layer
        """

        pass

    @abstractmethod
    def build_dense(self, layer):
        """Build spiking fully-connected layer.

        Parameters
        ----------

        layer: keras.layers.Layer
            Layer
        """

        pass

    @abstractmethod
    def build_convolution(self, layer):
        """Build spiking convolutional layer.

        Parameters
        ----------

        layer: keras.layers.Layer
            Layer
        """

        pass

    @abstractmethod
    def build_pooling(self, layer):
        """Build spiking pooling layer.

        Parameters
        ----------

        layer: keras.layers.Layer
            Layer
        """

        pass

    def build_flatten(self, layer):
        """Build flatten layer.

        May not be needed depending on the simulator.

        Parameters
        ----------

        layer: keras.layers.Layer
            Layer
        """

        pass

    @abstractmethod
    def compile(self):
        """Compile the spiking network."""

        pass

    @abstractmethod
    def simulate(self, **kwargs):
        """
        Simulate a spiking network for a certain duration, and record any
        variables of interest (spike trains, membrane potentials, ...)

        Returns
        -------

        output_b_l_t: ndarray
            Array of shape (`batch_size`, `num_classes`, ``num_timesteps``),
            containing the number of output spikes of the neurons in the final
            layer, for each sample and for each time step during the simulation.
        """

        pass

    @abstractmethod
    def reset(self, sample_idx):
        """Reset network variables.

        Parameters
        ----------

        sample_idx: int
            Index of sample that has just been simulated. In certain
            applications (video data), we may want to turn off reset between
            samples.
        """

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

    def init_cells(self):
        """
        Set cellparameters of neurons in each layer and initialize membrane
        potential.
        """

        pass

    def get_spiketrains(self, **kwargs):
        """Get spike trains of a layer.

        Returns
        -------

        spiketrains_b_l_t : ndarray
            Spike trains.
        """

        pass

    def get_spiketrains_input(self):
        """Get spike trains of input layer.

        Returns
        -------

        spiketrains_b_l_t : ndarray
            Spike trains of input.
        """

        pass

    def get_vmem(self, **kwargs):
        """Get membrane potentials of a layer.

        Returns
        -------

        mem_b_l_t : ndarray
            Membrane potentials of layer.
        """

        pass

    def build(self, parsed_model):
        """Assemble a spiking neural network to prepare for simulation.

        Parameters
        ----------

        parsed_model: keras.models.Model
            Parsed input model.
        """

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
        if self.fanout is None:
            self.set_connectivity()
            self.operations_ann = get_ann_ops(self.num_neurons,
                                              self.num_neurons_with_bias,
                                              self.fanin)
            print("Number of operations of ANN: {}".format(self.operations_ann))
            print("Number of neurons: {}".format(sum(self.num_neurons[1:])))
            print("Number of synapses: {}\n".format(self.num_synapses))

        self.is_built = True

    def run(self, x_test=None, y_test=None, dataflow=None, **kwargs):
        """ Simulate a spiking network.

        This methods takes care of preparing the dataset for batch-wise
        processing, and allocates variables for quantities measured during the
        simulation. The `simulate` method (overwritten by a concrete target
        simulator) is responsible for actually simulating the network over a
        given duration, and reporting the measured quantities. The `run` method
        then deals with evaluating this data to print statistics, plot figures,
        etc.

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
            Optional keyword arguments, for instance

            - path: Optional[str]
                Where to store the output plots. If no path given, this value is
                taken from the settings dictionary.

        Returns
        -------

        top1acc_total: float
            Number of correctly classified samples divided by total number of
            test samples.
        """

        from snntoolbox.conversion.utils import get_activations_batch
        from snntoolbox.utils.utils import in_top_k
        import snntoolbox.simulation.plotting as snn_plt

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
            from snntoolbox.datasets.aedat.DVSIterator import DVSIterator
            batch_shape = list(self.parsed_model.layers[0].batch_input_shape)
            batch_shape[0] = self.batch_size
            dvs_gen = DVSIterator(
                self.config['paths']['dataset_path'], batch_shape,
                self.config.getint('input', 'eventframe_width'),
                self.config.getint('input', 'num_dvs_events_per_sample'),
                eval(self.config['input']['chip_size']), batch_shape[2:],
                eval(self.config['input']['label_dict']))
            data_batch_kwargs['dvs_gen'] = dvs_gen

        # Simulate the SNN on a batch of samples in parallel.
        for batch_idx in range(num_batches):

            # Get a batch of samples
            x_b_l = None
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
            guesses_b_t = np.argmax(output_b_l_t, 1)
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

    def adjust_batchsize(self):
        """Reduce batch size to single sample if necessary.

        Not every simulator is able to simulate multiple samples in parallel.
        If this is the case (indicated by `is_parallelizable`), set `batch_size`
        to 1.
        """

        self._batch_size = self.config.getint('simulation', 'batch_size')
        if self._batch_size > 1 and not self.is_parallelizable:
            self.config.set('simulation', 'batch_size', '1')
            print("Temporarily setting batch_size to 1 because simulator does "
                  "not support parallel testing of multiple samples.")
            return 1
        return self._batch_size

    def restore_snn(self):
        """Restore both the spiking and the parsed network from disk.

        This method works for spiking Keras models
        `target_simulators.INI_target_sim`.
        """

        import keras
        print("Restoring spiking network...\n")
        self.load(self.config['paths']['path_wd'],
                  self.config['paths']['filename_snn'])
        self.parsed_model = keras.models.load_model(os.path.join(
            self.config['paths']['path_wd'],
            self.config['paths']['filename_parsed_model'] + '.h5'))

    def init_log_vars(self):
        """Initialize variables to record during simulation."""

        if 'input_b_l_t' in self._log_keys:
            self.input_b_l_t = np.empty(list(self.parsed_model.input_shape) +
                                        [self._num_timesteps], np.bool)

        if any({'spiketrains', 'spikerates', 'correlation', 'spikecounts',
                'hist_spikerates_activations'} & self._plot_keys) \
                or 'spiketrains_n_b_l_t' in self._log_keys:
            self.spiketrains_n_b_l_t = []
            for layer in self.parsed_model.layers:
                if not is_spiking(layer, self.config):
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
                if not is_spiking(layer, self.config):
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
        in the network. This includes e.g. the members `fanin`, `fanout`,
        `num_neurons`, `num_neurons_with_bias`.
        """

        from snntoolbox.parsing.utils import get_fanin, get_fanout

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
            if is_spiking(layer, self.config):
                self.fanin.append(get_fanin(layer))
                self.fanout.append(get_fanout(layer, self.config))
                self.num_neurons.append(np.prod(layer.output_shape[1:]))
                if hasattr(layer, 'b') and any(layer.b.get_value()):
                    print("Detected layer with biases: {}".format(layer.name))
                    self.num_neurons_with_bias.append(self.num_neurons[-1])
                else:
                    self.num_neurons_with_bias.append(0)

        self.num_synapses = np.dot(self.num_neurons, self.fanout)

        return self.num_neurons, self.num_neurons_with_bias, self.fanin

    def get_recorded_vars(self, layers):
        """Retrieve neuron variables recorded during simulation.

        If recorded, spike trains and membrane potentials will be inserted into
        the respective class members `spiketrains_n_b_l_t` and `mem_n_b_l_t`.
        In any case, this function must return an array containing the output
        spikes for each sample and time step.

        Parameters
        ----------

        layers
            List of SNN layers.

        Returns
        -------

        output_b_l_t: ndarray
            The output spikes.
            Shape: (`batch_size`, ``layer_shape``, ``num_timesteps``)

        """

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
                        spiketrains_b_l_t[b, l, :t+1])
        return output_b_l_t

    def reset_container_counters(self):
        self._mem_container_counter = 0
        self._spiketrains_container_counter = 0

    def set_mem_stats(self, mem):
        """Write recorded membrane potential out and plot it."""

        from snntoolbox.simulation.plotting import plot_potential

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
        """
        Count number of operations based on the input spike activity during
        simulation.
        """

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
        """
        Count number of operations based on the spike activity during
        simulation.

        Parameters
        ----------

        spiketrains_b_l_t: ndarray
            A batch of spikes for a layer over the simulation time.
            Shape: (`batch_size`, ``layer_shape``, ``num_timesteps``)

        """

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
        array into original layer ``shape``.

        Parameters
        ----------

        spiketrains: list
            List of spike times.
        shape
            Layer shape.

        Returns
        -------

        spiketrains_b_l_t: ndarray
            A batch of spikes for a layer over the simulation time.
            Shape: (`batch_size`, ``shape``, ``num_timesteps``)
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
    ``config['input']['sample_idxs_to_test']``, this function extracts them from
    the test set.
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
    """Build convolution layer.

    Parameters
    ----------

    layer: keras.layers.Conv2D
        Parsed model layer.
    delay: float
        Synaptic delay.

    Returns
    -------

    connections: List[tuple]
        A list where each entry is a tuple containing the source neuron index,
        the target neuron index, the connection strength (weight), and the
        synaptic ``delay``.
    i_offset: ndarray
        Flattened array containing the biases of all neurons in the ``layer``.
    """

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
    """Build average pooling layer.

    Parameters
    ----------

    layer: keras.layers.Pool2D
        Parsed model layer.
    delay: float
        Synaptic delay.

    Returns
    -------

    connections: List[tuple]
        A list where each entry is a tuple containing the source neuron index,
        the target neuron index, the connection strength (weight), and the
        synaptic ``delay``. The weight is given by :math:`\\frac{1}{k_x k_y}`,
        where :math:`k_x, k_y` are the dimensions of the pooling kernel.
    """

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


def spikecounts_to_rates(spikecounts_n_b_l_t):
    """Convert spiketrains to spikerates.

        The output will have the same shape as the input except for the last
        dimension, which is removed by replacing a sequence of spiketimes by a
        single rate value.

        Parameters
        ----------

        spikecounts_n_b_l_t: list[tuple[np.array, str]]

        Returns
        -------

        : list[tuple[np.array, str]]
            spikerates_n_b_l
        """

    t = spikecounts_n_b_l_t[0][0].shape[-1] + 1

    return [(np.true_divide(spikecounts_b_l_t[Ellipsis, -1], t), name)
            for (spikecounts_b_l_t, name) in spikecounts_n_b_l_t]


def spiketrains_to_rates(spiketrains_n_b_l_t, duration):
    """Convert spiketrains to spikerates.

    The output will have the same shape as the input except for the last
    dimension, which is removed by replacing a sequence of spiketimes by a
    single rate value.

    Parameters
    ----------

    spiketrains_n_b_l_t: list[tuple[np.array, str]]

    duration: int
        Duration of simulation.

    Returns
    -------

    spikerates_n_b_l: list[tuple[np.array, str]]
    """

    spikerates_n_b_l = []
    for (i, sp) in enumerate(spiketrains_n_b_l_t):
        shape = sp[0].shape[:-1]  # output_shape of layer
        # Allocate list containing an empty array of shape
        # 'output_shape' for each layer of the network, which will
        # hold the spikerates of a mini-batch.
        spikerates_n_b_l.append((np.empty(shape), sp[1]))
        # Count number of spikes fired in the layer and divide by the
        # simulation time to get the mean firing rate of each neuron.
        if len(shape) == 2:
            for ii in range(len(sp[0])):
                for jj in range(len(sp[0][ii])):
                    spikerates_n_b_l[i][0][ii, jj] = (
                        np.count_nonzero(sp[0][ii, jj]) / duration)
                    spikerates_n_b_l[i][0][ii, jj] *= np.sign(
                        np.sum(sp[0][ii, jj]))  # For negative spikes
        elif len(shape) == 4:
            for ii in range(len(sp[0])):
                for jj in range(len(sp[0][ii])):
                    for kk in range(len(sp[0][ii, jj])):
                        for ll in range(len(sp[0][ii, jj, kk])):
                            spikerates_n_b_l[i][0][ii, jj, kk, ll] = (
                                np.count_nonzero(sp[0][ii, jj, kk, ll]) /
                                duration)
                            spikerates_n_b_l[i][0][ii, jj, kk, ll] *= np.sign(
                                np.sum(sp[0][ii, jj, kk, ll]))

    return spikerates_n_b_l


def get_sample_activity_from_batch(activity_batch, idx=0):
    """Return layer activity for sample ``idx`` of an ``activity_batch``.
    """

    return [(layer_act[0][idx], layer_act[1]) for layer_act in activity_batch]


def get_spiking_outbound_layers(layer, config):
    """Iterate until spiking outbound layers are found.

    Parameters
    ----------

    layer: keras.layers.Layer
        Layer
    config: configparser.ConfigParser
        Settings.

    Returns
    -------

    : list
        List of outbound layers.
    """

    from snntoolbox.parsing.utils import get_outbound_layers

    outbound = layer
    while True:
        outbound = get_outbound_layers(outbound)
        if len(outbound) == 1:
            outbound = outbound[0]
            if is_spiking(outbound, config):
                return [outbound]
        else:
            result = []
            for outb in outbound:
                if is_spiking(outb, config):
                    result.append(outb)
                else:
                    result += get_spiking_outbound_layers(outb, config)
            return result


def get_layer_ops(spiketrains_b_l, fanout, num_neurons_with_bias=0):
    """
    Return total number of operations in the layer for a batch of samples.

    Parameters
    ----------

    spiketrains_b_l: ndarray
        Batch of spiketrains of a layer. Shape: (batch_size, layer_shape)
    fanout: int
        Number of outgoing connections per neuron.
    num_neurons_with_bias: int
        Number of neurons with bias.

    Returns
    -------

    layer_ops: int
        The total number of operations in the layer for a batch of samples.
    """

    return np.array([np.count_nonzero(s) for s in spiketrains_b_l]) * fanout + \
        num_neurons_with_bias


def get_ann_ops(num_neurons, num_neurons_with_bias, fanin):
    """
    Compute number of operations performed by an ANN in one forward pass.

    Parameters
    ----------

    num_neurons: list[int]
        Number of neurons per layer, starting with input layer.
    num_neurons_with_bias: list[int]
        Number of neurons with bias.
    fanin: list[int]
        List of fan-in of neurons in Conv, Dense and Pool layers. Input and Pool
        layers have fan-in 0 so they are not counted.


    Returns
    -------

    : int
        Number of operations.

    """

    return 2 * np.dot(num_neurons, fanin) + np.sum(num_neurons_with_bias)


def is_spiking(layer, config):
    """Test if layer is going to be converted to a layer that spikes.

    Parameters
    ----------

    layer: Keras.layers.Layer
        Layer of parsed model.
    config: configparser.ConfigParser
        Settings.

    Returns
    -------

    : bool
        ``True`` if converted layer will have spiking neurons.
    """

    return get_type(layer) in eval(config['restrictions']['spiking_layers'])
