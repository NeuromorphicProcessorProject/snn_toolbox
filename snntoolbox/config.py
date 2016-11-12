# coding=utf-8

"""Manage Parameters of SNNToolbox.

In the GUI, the toolbox settings are grouped in three categories:
    1. Global parameters ``globalparams``, specifying global settings for
       loading / saving, and what steps of the workflow to include (evaluation,
       normalization, conversion, simulation, ...)
    2. Neuron cell parameters ``cellparams``, determining properties of the
       spiking neurons (e.g. threshold, refractory period, ...). Not all of
       them are used in all simulators. For instance, our own simulator
       ``'INI'`` only uses a threshold, reset and membrane time constant.
    3. Simulation parameters ``simparams``, specifying for instance length and
       time resolution of the simulation run.

Parameters
..........

Global Parameters
*****************

dataset_path: string
    Select a directory where the toolbox will find the samples to test.
    Two input formats are supported:

        A) ``.npz``: Compressed numpy format.
        B) ``.jpg``: Images in directories corresponding to their class.

    A) Provide at least two compressed numpy files called ``x_test.npz``
    and ``y_test.npz`` containing the testset and groundtruth. In
    addition, if the network should be normalized, put a file
    ``x_norm.npz`` in the folder. This can be a the training set x_train,
    or a subset of it. Take care of memory limitations: If numpy can
    allocate a 4 GB float32 container for the activations to be
    computed during normalization, x_norm should contain not more than
    4*1e9*8bit/(fc*fx*fy*32bit) = 1/n samples, where (fc, fx, fy) is
    the shape of the largest layer, and n = fc*fx*fy its total cell
    count.

    B) The images are stored in subdirectories of the selected
    ``dataset_path``, where the names of the subdirectories represent
    their class label. The toolbox will then use Keras
    ``ImageDataGenerator`` to load and process the files batchwise.

dataset_format: string
    Two input formats are supported:

    - ``.npz``: Compressed numpy format.
    - ``.jpg``: Images in directories corresponding to their class.

datagen_kwargs: string, optional
    Specify keyword arguments for the data generator that will be used to load
    image files from subdirectories in the ``dataset_path``. Need to be given
    in form of a python dictionary.
    See ``keras.preprocessing.image.ImageDataGenerator`` for possible values.

dataflow_kwargs: string, optional

    Specify keyword arguments for the data flow that will get the samples from
    the ``ImageDataGenerator``. Need to be given in form of a python
    dictionary.
    See ``keras.preprocessing.image.ImageDataGenerator.flow_from_directory``
    for possible values.

model_lib: string
    The neural network library used to build the ANN, e.g.

    - 'keras'
    - 'lasagne'
    - 'caffe'

path_wd: string, optional
    Working directory. There, the toolbox will look for ANN models to convert
    or SNN models to test, load the parameters it needs and store (normalized)
    parameters.
    If not specified, the toolbox will use as destination for all files it
    needs to load and save:
    ``~/.snntoolbox/data/<filename_ann>/<simulator>/``.
    For instance, if we give ``'98.29'`` as filename of the ANN model to load,
    and use default parameters otherwise, the toolbox will perform all
    io-operations in ``~/.snntoolbox/data/mnist/mlp/98.29/INI/``.
log_dir_of_current_run: string, optional
    Path to directory where the output plots are stored. If not specified, will
    be ``<path_wd>/log/gui/<runlabel>``. ``<runlabel>`` can be specified in the
    GUI. Will be set to 'test' if None.
filename_ann: string
    Base name of all loaded and saved files during this run. The ANN model to
    be converted is expected to be named '<basename>'.
filename_parsed_model: string, optional
    Name given to parsed SNN models. If not specified by the user, the
    toolbox sets it to '<basename>_parsed'.
filename_snn: string, optional
    Name given to converted spiking nets when exported to test it in a specific
    simulator. If not specified by the user, the toolbox set it to
    ``snn_<basename>_<simulator>``.
evaluateANN: boolean, optional
    If enabled, test the input model before and after it is parsed, to ensure
    we do not lose performance. (Parsing extracts all necessary information
    from the input model and creates a new network with some simplifications in
    preparation for conversion to SNN.)
    If you also enabled 'normalization' (see parameter ``normalize`` below),
    then the network will be evaluated again after normalization. This
    operation should preserve accuracy as well.
normalize: boolean, optional
    Only relevant when converting a network, not during simulation. If enabled,
    the parameters of the spiking network will be normalized by the highest
    activation value, or by the ``n``-th percentile (see parameter
    ``percentile`` below).
percentile: int, optional
    Use the activation value in the specified percentile for normalization.
    Set to ``50`` for the median, ``100`` for the max. Typical values are
    ``99, 99.9, 100``.
convert: boolean, optional
    If enabled, load an ANN from ``<path_wd>`` and convert it to spiking.
simulate: boolean, optional
    If enabled, try to load SNN from ``<path_wd>`` and test it on the specified
    simulator (see parameter ``simulator``).
overwrite: boolean, optional
    If disabled, the save methods will ask for permission to overwrite files
    before writing parameters, activations, models etc. to disk.
batch_size: int, optional
    If the builtin simulator 'INI' is used, the batch size specifies
    the number of test samples that will be simulated in parallel. Important:
    When using 'INI' simulator, the batch size can only be run usingthe batch
    size it has been converted with. To run it with a different batch size,
    convert the ANN from scratch.
verbose: int, optional
    0: No intermediate results or status reports.
    1: Print progress of simulation and intermediate results.
    2: Record spiketrains of all layers for one sample, and save various plots
    (spiketrains, spikerates, activations, correlations, ...)
    3: Record, plot and return the membrane potential of all layers for the
    last test sample. Very time consuming. Works only with pyNN simulators.
scaling_factor: int, optional
    Used by the MegaSim simulator to scale the neuron parameters and weights
    because MegaSim uses integers.

Cell Parameters
***************

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
softmax_clockrate: int, optional
    In our implementation of a spiking softmax activation function we use an
    external Poisson clock to trigger calculating the softmax of a layer. The
    'softmax_clockrate' parameter sets the firing rate in Hz of this external
    clock. Note that this rate is limited by the maximum firing rate supported
    by the simulator (given by the inverse time resolution 1000 * 1 / dt Hz).

Simulation Parameters
*********************

simulator: string, optional
    Simulator with which to run the converted spiking network.
duration: float, optional
    Runtime of simulation of one input in milliseconds.
dt: float, optional
    Time resolution of spikes in milliseconds.
delay: float, optional
    Delay in milliseconds. Must be equal to or greater than the resolution.
poisson_input: float, optional
    If enabled, the input samples will be converted to Poisson spiketrains. The
    probability for a input neuron to fire is proportional to the analog value
    of the corresponding pixel, and limited by the parameter 'input_rate'
    below. For instance, with an 'input_rate' of 700, a fully-on pixel will
    elicit a Poisson spiketrain of 700 Hz. Turn off for a less noisy
    simulation. Currently, turning off Poisson input is only possible in INI
    simulator.
reset: string, optional
    Choose the reset mechanism to apply after spike.
    Reset to zero: After spike, the membrane potential is set to the resting
    potential.
    Reset by subtraction: After spike, the membrane potential is reduced by a
    value equal to the threshold.
input_rate: float, optional
    Poisson spike rate in Hz for a fully-on pixel of the input image. Note that
    the input_rate is limited by the maximum firing rate supported by the
    simulator (given by the inverse time resolution 1000 * 1 / dt Hz).
normalization_schedule: boolean, optional
    Reduce the normalization factor each layer.
online_normalization: boolean, optional
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
num_to_test: int, optional
    How many samples to test.
maxpool_type: string
    Implementation variants of spiking MaxPooling layers, based on
    fir_max: accumulated absolute firing rate
    avg_max: moving average of firing rate
binarize_weights: boolean, optional
    If ``True``, the weights are binarized.

Default values
..............

::

    globalparams = {'dataset_path': '',
                    'dataset_format': 'npz',
                    'datagen_kwargs': {},
                    'dataflow_kwargs': {},
                    'model_lib': 'keras',
                    'path_wd': '',
                    'log_dir_of_current_run': '',
                    'filename_ann': '',
                    'filename_parsed_model': '',
                    'filename_snn': '',
                    'batch_size': 100,
                    'evaluateANN': True,
                    'normalize': True,
                    'percentile': 99,
                    'convert': True,
                    'overwrite': True,
                    'simulate': True,
                    'verbose': 3}
    cellparams = {'v_thresh': 1,
                  'v_reset': 0,
                  'v_rest': 0,
                  'e_rev_E': 10,
                  'e_rev_I': -10,
                  'i_offset': 0,
                  'cm': 0.09,
                  'tau_m': 1000,
                  'tau_refrac': 0,
                  'tau_syn_E': 0.01,
                  'tau_syn_I': 0.01,
                  'softmax_clockrate': 300}
    simparams = {'simulator': 'INI',
                 'duration': 200,
                 'dt': 1,
                 'delay': 1,
                 'poisson_input': False,
                 'reset': 'Reset by subtraction',
                 'input_rate': 1000,
                 'timestep_fraction': 10,
                 'diff_to_max_rate': 200,
                 'num_to_test': 1000,
                 'diff_to_min_rate': 100,
                 'binarize_weights': False}
"""

import matplotlib as mpl
from textwrap import dedent

# Define text sizes of various plot properties relative to font.size, using the
# following values: xx-small, x-small, small, medium, large, x-large, xx-large,
# larger, or smaller.
plotproperties = {'font.size': 13,
                  'axes.titlesize': 'xx-large',
                  'axes.labelsize': 'x-large',
                  'xtick.labelsize': 'x-large',
                  'xtick.major.size': 7,
                  'xtick.minor.size': 5,
                  'ytick.labelsize': 'x-large',
                  'ytick.major.size': 7,
                  'ytick.minor.size': 5,
                  'legend.fontsize': 'x-large',
                  'figure.figsize': (7, 6),
                  'savefig.format': 'png'}
mpl.rcParams.update(plotproperties)

# List supported model libraries, simulators, etc.
model_libs = {'keras', 'lasagne', 'caffe'}
maxpool_types = {'fir_max', 'exp_max', 'avg_max', 'binary_tanh',
                 'binary_sigmoid'}
simulators_pyNN = {'nest', 'brian', 'Neuron'}
simulators_other = {'INI', 'brian2', 'MegaSim'}
simulators = simulators_pyNN.copy()
simulators.update(simulators_other)

# Default parameters:
settings = {'dataset_path': 'dataset/',
            'dataset_format': 'npz',
            'datagen_kwargs': '{}',
            'dataflow_kwargs': '{}',
            'model_lib': 'keras',
            'path_wd': '',
            'log_dir_of_current_run': '',
            'filename_parsed_model': '',
            'filename_ann': '',
            'filename_snn': '',
            'batch_size': 100,
            'samples_to_test': '',
            'evaluateANN': True,
            'normalize': True,
            'percentile': 99,
            'overwrite': True,
            'convert': True,
            'simulate': True,
            'verbose': 3,
            'v_thresh': 1,
            'tau_refrac': 0,
            'softmax_clockrate': 300,
            'simulator': 'INI',
            'duration': 200,
            'dt': 1,
            'num_to_test': 1000,
            'poisson_input': False,
            'reset': 'Reset by subtraction',
            'input_rate': 1000,
            'normalization_schedule': False,
            'online_normalization': False,
            'payloads': False,
            'diff_to_max_rate': 200,
            'timestep_fraction': 10,
            'diff_to_min_rate': 100,
            'scaling_factor': 10000000,
            'maxpool_type': 'fir_max',
            'binarize_weights': False}

# pyNN specific parameters.
pyNN_settings = {'v_reset': 0,
                 'v_rest': 0,  # Initial neuron potential
                 'e_rev_E': 10,  # Pull v towards +10 mV during spike
                 'e_rev_I': -10,  # Pull v towards -10 mV during spike
                 'i_offset': 0,  # No offset current
                 'cm': 0.09,  # Fast integration. Small cm slows down sim.
                 'tau_m': 1000,  # No leakage
                 'tau_syn_E': 0.01,  # Excitatory synaptic cond decays fast
                 'tau_syn_I': 0.01,  # Inhibitory synaptic cond decays fast
                 'delay': 1}  # Constraint: delay >= dt

# Merge settings
settings.update(pyNN_settings)

# Layers that can be implemented by our spiking neuron simulators
spiking_layers = {'Dense', 'Convolution2D', 'MaxPooling2D', 'AveragePooling2D',
                  'Flatten'}


def file_not_found_msg(filename, path=None):
    """Return string stating that file was not found at path.

    Parameters
    ----------
    filename: str
    path: Optional[str]

    Returns
    -------

    : str
    """

    if path is None:
        path = settings['path_wd']
    return "Input model file {} not found in working dir {}".format(filename,
                                                                    path)


def update_setup(s):
    """Update parameters.

    Check that parameter choices ``s`` are valid and update the global
    parameter settings ``snntoolbox.config.settings`` with the user-specified
    values. Default values are filled in where user did not give any.
    """
    import os
    import snntoolbox

    assert isinstance(s, dict), "Input argument must be a dictionary."

    # Update settings with user specified settings.
    settings.update(s)

    # Name of input file must be given.
    assert 'filename_ann' in s, "Filename of input model not specified."

    # Check that simulator choice is valid.
    assert settings['simulator'] in simulators, \
        "Simulator '{}' not supported.".format(settings['simulator']) + \
        " Choose from {}".format(simulators)

    # Warn user that it is not possible to use Brian2 simulator by loading a
    # pre-converted network from disk.
    if settings['simulator'] == 'brian2' and not settings['convert']:
        print(dedent("""\ \n
            SNN toolbox Warning: When using Brian 2 simulator, you need to
            convert the network each time you start a new session. (No
            saving/reloading methods implemented.) Setting convert = True.
            \n"""))
        settings['convert'] = True

    # Set default path if user passed empty string.
    if 'path_wd' in s and s['path_wd'] == '':
        settings['path_wd'] = os.path.join(snntoolbox.toolbox_root, 'data',
                                           settings['filename_ann'],
                                           settings['simulator'])

    # Check specified working directory exists.
    assert os.path.exists(settings['path_wd']), \
        "Working directory {} does not exist.".format(settings['path_wd'])

    # Check that choice of input model library is valid.
    assert settings['model_lib'] in model_libs, \
        "Input model library '{}' ".format(settings['model_lib']) + \
        "not supported yet. Possible values: {}".format(model_libs)

    # Check input model is found and has the right format for the specified
    # model library.
    if settings['model_lib'] == 'caffe':
        caffemodel_file = settings['filename_ann'] + '.caffemodel'
        assert os.path.isfile(
            os.path.join(settings['path_wd'], caffemodel_file)), \
            file_not_found_msg(caffemodel_file)
        prototxt_file = settings['filename_ann'] + '.prototxt'
        assert os.path.isfile(
            os.path.join(settings['path_wd'], prototxt_file)), \
            file_not_found_msg(prototxt_file)
    elif settings['model_lib'] == 'keras':
        h5_file = settings['filename_ann'] + '.h5'
        h5_filepath = os.path.join(settings['path_wd'], h5_file)
        assert os.path.isfile(h5_filepath), \
            file_not_found_msg(h5_file)
        json_file = settings['filename_ann'] + '.json'
        if not os.path.isfile(os.path.join(settings['path_wd'], json_file)):
            import keras
            try:
                keras.models.load_model(h5_filepath)
            except:
                raise AssertionError("You provided an h5 file with weights, "
                                     "but without network configuration. In "
                                     "earlier versions of Keras, this is "
                                     "contained in a json file.")
    elif settings['model_lib'] == 'lasagne':
        h5_file = settings['filename_ann'] + '.h5'
        assert os.path.isfile(os.path.join(settings['path_wd'], h5_file)), \
            file_not_found_msg(h5_file)
        py_file = settings['filename_ann'] + '.py'
        assert os.path.isfile(os.path.join(settings['path_wd'], py_file)), \
            file_not_found_msg(py_file)
    else:
        print("For the specified input model library {}, ".format(
            settings['model_lib']) + "no test is implemented to check if input "
            "model files exist in the specified working directory!")

    # Check that the data set path is valid and contains the data in the
    # specified format. For jpg format we only do a superficial check because
    # listing the subdirectory contents becomes too costly for large data sets.
    assert os.path.exists(settings['dataset_path']), \
        "Path to data set does not exist."
    if settings['normalize'] and settings['dataset_format'] == 'npz' and not \
            os.path.exists(
                os.path.join(settings['dataset_path'], 'x_norm.npz')):
        raise AssertionError(
            "No data set file 'x_norm.npz' found in specified data set path " +
            "{}. Add it, or disable normalization.".format(
                settings['dataset_path']))
    if settings['dataset_format'] == 'npz' and not (os.path.exists(os.path.join(
            settings['dataset_path'], 'x_test.npz')) and os.path.exists(
            os.path.join(settings['dataset_path'], 'y_test.npz'))):
        raise AssertionError(
            "Data set file 'x_test.npz' or 'y_test.npz' was not found in "
            "specified data set path {}.".format(
                settings['dataset_path']))
    if settings['dataset_format'] == 'jpg':
        assert os.listdir(settings['dataset_path']), "Data set directory is" \
                                                     " empty."

    # Convert string containing sample indices to list of indices.
    assert isinstance(settings['samples_to_test'], str), "The parameter " + \
        "'samples_to_test' must be of type 'string'. Given: {}".format(
        settings['samples_to_test'].__class__)
    settings['sample_indices_to_test'] = [
        int(i) for i in settings['samples_to_test'].split() if i.isnumeric()]

    # Create log directory if it does not exist.
    if 'log_dir_of_current_run' not in s:
        settings['log_dir_of_current_run'] = os.path.join(s['path_wd'], 'log',
                                                          'gui', 'test')
    if not os.path.isdir(settings['log_dir_of_current_run']):
        os.makedirs(settings['log_dir_of_current_run'])

    # Specify filenames for models at different stages of the conversion.
    if 'filename_parsed_model' not in s or s['filename_parsed_model'] == '':
        settings['filename_parsed_model'] = settings['filename_ann'] + '_parsed'
    if 'filename_snn' not in s or s['filename_snn'] == '':
        settings['filename_snn'] = '{}_{}'.format(settings['filename_ann'],
                                                  settings['simulator'])

    if settings['simulator'] != 'INI' and not settings['poisson_input']:
        settings['poisson_input'] = True
        print(dedent("""\
            SNN toolbox Warning: Currently, turning off Poisson input is
            only possible in INI simulator. Falling back on Poisson input."""))

    # Make sure the number of samples to test is not lower than the batch size.
    if settings['num_to_test'] < settings['batch_size']:
        print(dedent("""\
            SNN toolbox Warning: 'num_to_test' set lower than 'batch_size'.
            In simulators that test samples batch-wise (e.g. INIsim), this
            can lead to undesired behavior. Setting 'num_to_test' equal to
            'batch_size'."""))
        settings['num_to_test'] = settings['batch_size']

    return True


def initialize_simulator(simulator, **kwargs):
    """Import a module that contains utility functions of spiking simulator."""
    from importlib import import_module

    sim = None
    if simulator in simulators_pyNN:
        assert 'dt' in kwargs, "Need to pass timestep as keyword-argument" \
                               " 'dt=' if initializing a pyNN simulator."
        if simulator == 'nest':
            # Workaround for missing link bug, see
            # https://github.com/ContinuumIO/anaconda-issues/issues/152
            # noinspection PyUnresolvedReferences
            import readline
        sim = import_module('pyNN.' + simulator)

        # From the pyNN documentation:
        # "Before using any other functions or classes from PyNN, the user
        # must call the setup() function. Calling setup() a second time
        # resets the simulator entirely, destroying any network that may
        # have been created in the meantime."
        sim.setup(timestep=kwargs['dt'])
    elif simulator == 'brian2':
        sim = import_module('brian2')
    elif simulator == 'INI':
        sim = import_module('snntoolbox.core.inisim')
    elif simulator == 'MegaSim':
        sim = import_module('snntoolbox.core.megasim')
    assert sim, "Simulator {} could not be initialized.".format(simulator)
    print("Initialized {} simulator.\n".format(simulator))
    return sim
