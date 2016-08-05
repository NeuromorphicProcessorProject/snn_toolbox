"""
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
    Where to load the dataset from. Used for testing the network. Dataset needs
    to be in npy format.

model_lib: string
    The neural network library used to build the ANN, e.g.

    - 'keras'
    - 'lasagne'
    - 'caffe'

path: string, optional
    Working directory. There, the toolbox will look for ANN models to convert
    or SNN models to test, load the parameters it needs and store (normalized)
    parameters.
    If not specified, the toolbox will use as destination for all files it
    needs to load and save:
    ``~/.snntoolbox/data/<filename>/<simulator>/``.
    For instance, if we give ``'98.29'`` as filename of the ANN model to load,
    and use default parameters otherwise, the toolbox will perform all
    io-operations in ``~/.snntoolbox/data/mnist/mlp/98.29/INI/``.
log_dir_of_current_run: string, optional
    Path to directory where the output plots are stored. If not specified, will
    be ``<path>/log/gui/<runlabel>``. ``<runlabel>`` can be specified in the
    GUI. Will be set to 'test' if None.
filename: string
    Base name of all loaded and saved files during this run. The ANN model to
    be converted is expected to be named '<basename>'.
filename_snn: string, optional
    Name given to converted SNN models. If not specified by the user, the
    toolbox sets it to 'snn_<basename>'.
filename_snn_exported: string, optional
    Name given to converted spiking nets when exported to test it in a specific
    simulator. If not specified by the user, the toolbox set it to
    ``snn_<basename>_<simulator>``.
evaluateANN: boolean, optional
    If enabled, test the ANN before conversion. If you also enabled
    'normalization' (see parameter ``normalize`` below), then the network will
    be evaluated again after normalization.
normalize: boolean, optional
    Only relevant when converting a network, not during simulation. If enabled,
    the parameters of the spiking network will be normalized by the highest
    parameter or activation value.
percentile: int, optional
    Use the activation value in the specified percentile for normalization.
    Set to ``50`` for the median, ``100`` for the max.
convert: boolean, optional
    If enabled, load an ANN from ``<path>`` and convert it to spiking.
simulate: boolean, optional
    If enabled, try to load SNN from ``<path>`` and test it on the specified
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
diff_to_max_rate: float, optional
    The converted spiking network performs best if the average firing rates of
    each layer are not higher but also not much lower than the maximum rate
    supported by the simulator (inverse time resolution). Normalization
    eliminates saturation but introduces undersampling (parameters are
    normalized with respect to the highest value in a batch). To overcome this,
    the spikerates of each layer are monitored during simulation. If they drop
    below the maximum firing rate by more than 'diff to max rate', we divide
    the parameters of the layer by its highest rate. Set the parameter in Hz.
timestep_fraction: int, optional
    If set to 10 (default), the parameter modification mechanism described in
    'diff_to_max_rate' will be performed at every 10th timestep.
min_rate: float, optional
    Minimum spikerate in Hz. When The firing rates of a layer are below this
    value, the weights will NOT be modified in the feedback mechanism described
    in 'diff_to_max_rate'. This is useful in the beginning of a simulation,
    when higher layers need some time to integrate up a sufficiently high
    membrane potential.
num_to_test: int, optional
    How many samples to test.

Default values
..............

::

    globalparams = {'dataset_path': '',
                    'model_lib': 'keras',
                    'path': '',
                    'log_dir_of_current_run': '',
                    'filename': '',
                    'filename_snn': 'snn_',
                    'filename_snn_exported': 'snn_exported_',
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
                 'num_to_test': 10,
                 'min_rate': 100}

"""

import matplotlib as mpl

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
                  'figure.figsize': (7, 6)}
mpl.rcParams.update(plotproperties)

# List supported model libraries, simulators, etc.
model_libs = {'keras', 'lasagne', 'caffe'}
simulators_pyNN = {'nest', 'brian', 'Neuron'}
simulators_other = {'INI', 'brian2'}
simulators = simulators_pyNN.copy()
simulators.update(simulators_other)

# Default parameters:
settings = {'dataset_path': '',
            'model_lib': 'keras',
            'path': '',
            'log_dir_of_current_run': '',
            'filename': '',
            'filename_snn': 'snn_',
            'filename_snn_exported': 'snn_exported_',
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
            'poisson_input': False,
            'reset': 'Reset by subtraction',
            'input_rate': 1000,
            'diff_to_max_rate': 200,
            'timestep_fraction': 10,
            'min_rate': 100}

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
                 'delay': 1,  # Constraint: delay >= dt
                 'num_to_test': 10}

# Layers followed by an Activation layer
activation_layers = {'Dense', 'Convolution2D'}

bn_layers = {'Dense', 'Convolution2D'}


def update_setup(s=None):
    """
    Update parameters

    Check that parameter choices ``s`` are valid and update the global
    parameter settings ``snntoolbox.config.settings`` with the user-specified
    values. Default values are filled in where user did not give any.
    """

    import os
    import snntoolbox

    if s is None:
        s = {}

    # Check that choice of input model library is valid (not really needed when
    # using GUI because options are hardwired in dropdown list).
    if 'model_lib' in s:
        assert s['model_lib'] in model_libs, \
            "Input model library '{}' ".format(s['model_lib']) + \
            "not supported yet. Possible values: {}".format(model_libs)
    # Name of input file must be given.
    assert 'filename' in s, "Filename of stored model not specified."
    # Check that simulator choice is valid (not really needed when using GUI
    # because options are hardwired in dropdown list).
    if 'simulator' in s:
        assert s['simulator'] in simulators, \
            "Simulator '{}' not supported.".format(s['simulator']) + \
            " Choose from {}".format(simulators)
    else:
        # Fall back on default if none specified.
        s.update({'simulator': 'INI'})
    # Warn user that it is not possible to use Brian2 simulator by loading a
    # pre-converted network from disk.
    if s['simulator'] == 'brian2' and 'convert' in s and not s['convert'] \
            or 'convert' not in s and not settings['convert']:
        print('\n')
        print("""SNN toolbox Warning: When using Brian 2 simulator, you need to
              convert the network each time you start a new session. (No
              saving/reloading methods implemented.)
              Setting convert = True.\n""")
        s['convert'] = True
    # Set default path if user did not specify one.
    if 'path' not in s or s['path'] == '':
        s['path'] = os.path.join(snntoolbox._dir, 'data', s['filename'],
                                 s['simulator'])
    # Create directory if not there yet.
    if not os.path.exists(s['path']):
        os.makedirs(s['path'])

    # Convert string containing sample indices to list of indices.
    if 'samples_to_test' not in s:
        s['samples_to_test'] = ''
    s['sample_indices_to_test'] = [
        int(i) for i in s['samples_to_test'].split() if i.isnumeric()]

    if 'log_dir_of_current_run' not in s:
        s['log_dir_of_current_run'] = os.path.join(s['path'], 'log', 'gui',
                                                   'test')

    # Specify filenames for models at different stages of the conversion.
    s['filename_snn'] = 'snn_' + s['filename']
    s['filename_snn_exported'] = s['filename_snn'] + '_' + s['simulator']

    if s['simulator'] != 'INI' and not s['poisson_input']:
        s['poisson_input'] = True
        print("""SNN toolbox Warning: Currently, turning off Poisson input is
            only possible in INI simulator. Falling back on Poisson input.""")

    # If there are any parameters specified, merge with default parameters.
    settings.update(s)


def initialize_simulator(simulator=None):
    """ Import module containing utility functions of spiking simulator"""
    from importlib import import_module

    if simulator is None:
        simulator = settings['simulator']

    if simulator in simulators_pyNN:
        if simulator == 'nest':
            # Workaround for missing link bug, see
            # https://github.com/ContinuumIO/anaconda-issues/issues/152
            import readline
        sim = import_module('pyNN.' + simulator)
        # From the pyNN documentation:
        # "Before using any other functions or classes from PyNN, the user
        # must call the setup() function. Calling setup() a second time
        # resets the simulator entirely, destroying any network that may
        # have been created in the meantime."
        sim.setup(timestep=settings['dt'])
    elif simulator == 'brian2':
        sim = import_module('brian2')
    elif simulator == 'INI':
        sim = import_module('snntoolbox.core.inisim')

    print("Initialized {} simulator.\n".format(simulator))
    return sim
