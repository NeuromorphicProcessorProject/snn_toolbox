.. # coding=utf-8

.. _configuration:

Configuration
=============

To configure the toolbox for a specific experiment, create a plain text file
and add the parameters you want to set, using `INI/conf file syntax <https://docs.python.org/3/library/configparser.html>`_.
See also our :doc:`examples`. Any settings you do not specify will be filled in
with the :ref:`default values <default-settings>`. When
:doc:`starting the toolbox <running>`, you may pass the location of this
settings file as argument to the program.

The toolbox settings are grouped in the following categories:

[paths]
-------

path_wd: str, optional
    Working directory. There, the toolbox will look for ANN models to convert
    or SNN models to test. If not specified, the toolbox will try to create and
    use the directory ``~/.snntoolbox/data/<filename_ann>/<simulator>/``.

dataset_path: str
    Select a directory where the toolbox will find the samples to test.
    See ``dataset_format`` for supported input types.

log_dir_of_current_run: str, optional
    Path to directory where the output plots and logs are stored. Default:
    ``<path_wd>/log/gui/<runlabel>``.

runlabel: str, optional
    Label of current experiment. Default: 'test'.

filename_ann: str
    Name of ANN model to be converted.

filename_parsed_model: string, optional
    Name given to parsed SNN model. Default: '<filename_ann>_parsed'.

filename_snn: str, optional
    Name given to converted spiking model when exported to test it in a specific
    simulator. Default: ``<filename_ann>_<simulator>``.

filename_clamp_indices: str, optional
    Name of file containing a dictionary of clamp indices. Each ``key``
    specifies a layer index, and the corresponding ``value`` defines the number
    of time steps during which the membrane potential of neurons in this layer
    are clamped to zero. If this option is not specified, no layers are clamped.

class_idx_path: str, optional
    Only needed if the data set is stored as images in folders denoting the
    class label (i.e. ``dataset_format = jpg`` below). Then ``class_idx_path``
    is the path to a file containing a dictionary that maps the class labels to
    the corresponding indices of neurons in the output layer.

[input]
-------

model_lib: str
    The neural network library used to build the ANN. Currently supported:

    - ``keras``
    - ``lasagne``
    - ``caffe``

dataset_format: str, optional
    The following input formats are supported:

    A) ``npz``: Compressed numpy format.
    B) ``jpg``: Images in directories corresponding to their class.
    C) ``aedat``: Sequence of address-events recorded from a Dynamic Vision Sensor.

    A) Default. Provide at least two compressed numpy files called ``x_test.npz``
    and ``y_test.npz`` containing the test set and ground truth. In
    addition, if the network should be normalized, put a file
    ``x_norm.npz`` in the folder. This can be a the training set, or a subset of
    it. Take care of memory limitations: If numpy can allocate a 4 GB float32
    container for the activations to be computed during normalization, ``x_norm``
    should contain not more than 4*1e9*8bit/(fc*fx*fy*32bit) = 1/n samples,
    where (fc, fx, fy) is the shape of the largest layer, and n = fc*fx*fy its
    total cell count.

    B) The images are stored in subdirectories of the selected
    ``dataset_path``, where the names of the subdirectories represent their
    class label. The toolbox will then use
    ``keras.preprocessing.image.ImageDataGenerator`` to load and process the
    files batchwise. Setting ``jpg`` here works even if the images are actually
    in ``.png`` or ``.bmp`` format.

    C) Beta stage.

datagen_kwargs: str, optional
    Specify keyword arguments for the data generator that will be used to load
    image files from subdirectories in the ``dataset_path``. Need to be given
    in form of a python dictionary. See
    ``keras.preprocessing.image.ImageDataGenerator`` for possible values.

dataflow_kwargs: str, optional
    Specify keyword arguments for the data flow that will get the samples from
    the ``ImageDataGenerator``. Need to be given in form of a python dictionary.
    See ``keras.preprocessing.image.ImageDataGenerator.flow_from_directory`` for
    possible values.

poisson_input: float, optional
    If enabled, the input samples will be converted to Poisson spiketrains. The
    probability for a input neuron to fire is proportional to the analog value
    of the corresponding pixel, and limited by the parameter 'input_rate' below.
    For instance, with an ``input_rate`` of 200, a fully-on pixel will elicit a
    Poisson spiketrain of 200 Hz. Turn off for a less noisy simulation.
    Currently, turning off Poisson input is only possible in INI simulator.

input_rate: float, optional
    Poisson spike rate in Hz for a fully-on pixel of the input image. Note that
    the input_rate is limited by the maximum firing rate supported by the
    simulator (given by the inverse time resolution 1000 * 1 / dt Hz).

num_poisson_events_per_sample: int, optional
    Limits the number of Poisson spikes generated from each frame.
    Default: -1 (unlimited).

num_dvs_events_per_sample: int, optional
    Number of DVS events used in one image classification trial. Can be thought
    of as being equivalent to one frame. Default: 2000.

eventframe_width: int, optional
    To be able to use asynchronous DVS events in our time-stepped simulator, we
    collect them into frames (binary maps) which are presented to the SNN at
    subsequent time steps. The option ``eventframe_width`` defines how many
    timesteps the timestamps of events in such a frame should span at most.
    Default: 10.

label_dict: dict
    Dictionary containing the class labels. Only needed with ``.aedat`` input.

chip_size: tuple
    When using ``.aedat`` input, the addresses can be checked for outliers, or
    may have to be subsampled from the original ``chip_size`` to the image
    dimension required by the network. Set ``chip_size`` to the shape of the DVS
    chip that was used to record the aedat sample, e.g. (240, 180). The image
    dimension to subsample to will be infered from the shape of the input layer
    of the network.

frame_gen_method: str
    How to accumulate DVS events into frames.

        - ``signed_sum``: DVS events are added up while their polarity is taken
          into account. (ON and OFF events cancel each other out.)
        - ``rectified_sum``: Polarity is discarded; all events are considered ON.

is_x_first: bool
    Whether the x-address of a DVS events is considered as the first dimension
    when accumulating events into 2-D frame.

is_x_flipped: bool
    Whether to reflect DVS image through vertical axis.

is_y_flipped: bool
    Whether to reflect DVS image through horizontal axis.

[tools]
-------

evaluateANN: bool, optional
    If enabled, the ANN is tested at two stages:

        1. At the very beginning, using the input model as provided by the user.
        2. After parsing the input model to our internal Keras representation,
           and applying any optional modifications like replacing Softmax
           activation by ReLU, replacing MaxPooling by AveragePooling, and
           normalizing the network parameters.

    This ensures all operations on the ANN preserve the accuracy.

normalize: bool, optional
    If enabled, the parameters of each layer will be normalized by the highest
    activation value, or by the ``n``-th percentile (see parameter
    ``percentile`` below).

convert: bool, optional
    If enabled, load an ANN from ``path_wd`` and convert it to spiking.

simulate: bool, optional
    If enabled, load SNN from ``path_wd`` and test it on the specified
    simulator (see parameter ``simulator``).

[normalization]
---------------

percentile: int, optional
    Use the activation value in the specified percentile for normalization.
    Set to ``50`` for the median, ``100`` for the max. Typical values are
    ``99, 99.9, 100``.

normalization_schedule: bool, optional
    Reduce the normalization factor each layer.

online_normalization: bool, optional
    The converted spiking network performs best if the average firing rates of
    each layer are not higher but also not much lower than the maximum rate
    supported by the simulator (inverse time resolution). Normalization
    eliminates saturation but introduces undersampling (parameters are
    normalized with respect to the highest value in a batch). To overcome this,
    the spikerates of each layer are monitored during simulation. If they drop
    below the maximum firing rate by more than 'diff to max rate', we set the
    threshold of the layer to its highest rate.

diff_to_max_rate: float, optional
    If the highest firing rate of neurons in a layer drops below the maximum
    firing rate by more than 'diff to max rate', we set the threshold of the
    layer to its highest rate. Set the parameter in Hz.

diff_to_min_rate: float, optional
    When The firing rates of a layer are below this value, the weights will NOT
    be modified in the feedback mechanism described in 'online_normalization'.
    This is useful in the beginning of a simulation, when higher layers need
    some time to integrate up a sufficiently high membrane potential.

timestep_fraction: int, optional
    If set to 10 (default), the parameter modification mechanism described in
    'online_normalization' will be performed at every 10th timestep.

[conversion]
------------

softmax_to_relu: bool, optional
    If ``True``, replace softmax by ReLU activation function. This is
    recommended (default), because the spiking softmax implementation tends to
    reduce accuracy, especially top-5. It is safe to do this replacement as long
    as the input to the activation function is not all negative. In that case,
    the ReLU would not be able to determine the winner.

maxpool_type: str, optional
    Implementation variants of spiking MaxPooling layers, based on

        - ``fir_max``: accumulated absolute firing rate (default)
        - ``avg_max``: moving average of firing rate
        - ``exp_max``: exponential FIR filter.

max2avg_pool: bool, optional
    If ``True``, max pooling layers are replaced by average pooling.

spike_code: str, optional
    Describes the code used to transform analog activation values of the
    original network into spikes.

        - ``temporal_mean_rate`` (default): Average over number of spikes that
          occur during simulation ``duration``.
        - ``temporal_pattern``: Analog activation value is transformed into
          binary representation of spikes.
        - ``ttfs``: Instantaneous firing rate is given by the inverse
          time-to-first-spike.
        - ``ttfs_dyn_thresh``: Like ``ttfs``, but with a threshold that adapts
          dynamically to the amount of input a neuron has received.
        - ``ttfs_corrective``: Allows corrective spikes to be fired to improve
          the first guess made by ``ttfs``.

num_bits: int, optional
    Bit-resolution that a binary spike train can maximally encode when using
    ``spike_code = temporal_pattern``.

[simulation]
------------

simulator: str, optional
    Simulator with which to run the converted spiking network.

duration: float, optional
    Runtime of simulation of one input in milliseconds.

dt: float, optional
    Time resolution of spikes in milliseconds.

num_to_test: int, optional
    How many samples to test.

sample_idxs_to_test: Iterable, optional
    List of sample indices to test.

batch_size: int, optional
    If the builtin simulator 'INI' is used, the batch size specifies
    the number of test samples that will be simulated in parallel.

reset_between_nth_sample: int, optional
    When testing a video sequence, this option allows turning off the reset
    between individual samples. Default: 1 (reset after every frame). Set to a
    negative value to turn off reset completely.

top_k: int, optional
    In addition to the top-1 error, report ``top_k`` error during simulation.
    Default: 5.

keras_backend: str, optional
    The backend to use in ``INI`` simulator.

        - ``theano``: Only works in combination with
          ``spike_code = temporal_mean_rate``.
        - ``tensorflow``: Does not implement the spiking MaxPool layer when
          using ``spike_code = temporal_mean_rate``.

[cell]
------

v_thresh: float, optional
    Threshold in mV defining the voltage at which a spike is fired.

v_reset: float, optional
    Reset potential in mV of the neurons after spiking.

v_rest: float, optional
    Resting membrane potential in mV.

e_rev_E: float, optional
    Reversal potential for excitatory input in mV.

e_rev_I: float, optional
    Reversal potential for inhibitory input in mV.

i_offset: float, optional
    Offset current in nA.

cm: float, optional
    Membrane capacitance in nF.

tau_m: float, optional
    Membrane time constant in milliseconds.

tau_refrac: float, optional
    Duration of refractory period in milliseconds of the neurons after spiking.

tau_syn_E: float, optional
    Decay time of the excitatory synaptic conductance in milliseconds.

tau_syn_I: float, optional
    Decay time of the inhibitory synaptic conductance in milliseconds.

delay: float, optional
    Delay in milliseconds. Must be equal to or greater than the resolution.

binarize_weights: bool, optional
    If ``True``, the weights are binarized.

scaling_factor: int, optional
    Used by the MegaSim simulator to scale the neuron parameters and weights
    because MegaSim uses integers.

payloads: bool, optional
    Whether or not to send a float value together with each spike.

reset: str, optional
    Choose the reset mechanism to apply after spike.

        - 'Reset to zero': After spike, the membrane potential is set to the
          resting potential.
        - 'Reset by subtraction': After spike, the membrane potential is reduced
          by a value equal to the threshold.
        - 'Reset by modulo': After spike, the membrane potential is reduced by
          the smallest multiple of the threshold such that the new membrane
          potential is below threshold.

leak: bool, optional
   Experimental feature. ``False`` by default.

[parameter_sweep]
-----------------

Enables running the toolbox with the same settings except for one parameter
being varied. In beta stadium.

param_values: list, optional
    Contains the parameter values for which the simulation will be repeated.
param_name: str, optional
    Label indicating the parameter to sweep, e.g. ``'v_thresh'``.
param_logscale: bool, optional
    If ``True``, plot test accuracy vs ``params`` in log scale.

[output]
--------

log_vars: set, optional
    Specify the variables to monitor and save to disk. Possible values:
    'activations_n_b_l', 'spiketrains_n_b_l_t', 'input_b_l_t', 'mem_n_b_l_t',
    'synaptic_operations_b_t', 'neuron_operations_b_t', 'all'.
    Default: ``{}``.

plot_vars: set, optional
    Specify the variables to monitor and plot. Possible values:
    'activations', 'spiketrains', 'spikecounts', 'spikerates', 'input_image',
    'error_t', 'confusion_matrix', 'correlation', 'hist_spikerates_activations',
    'normalization_activations', 'operations', 'all'.
    Default: ``{}``.

verbose: int, optional
    If nonzero (default), print current error rate at every time step during
    simulation.

overwrite: bool, optional
    If ``False``, the save methods will ask for permission to overwrite files
    before writing parameters, activations, models etc. to disk. Default:
    ``True``.

plotproperties: dict, optional
    Options that modify matplotlib plot properties.


.. _default-settings:

SNN toolbox default settings
----------------------------

.. include:: ../../../snntoolbox/config_defaults
   :literal:
