"""
The toolbox settings are grouped in three categories:
    1. Global parameters ``globalparams``, specifying global settings for
       loading / saving, and what steps of the workflow to include (evaluation,
       normalization, conversion, simulation, ...)
    2. Neuron cell parameters ``cellparams``, determining properties of the
       spiking neurons (e.g. threshold, refractory period, ...). Not all of
       them are used in all simulators. For instance, our own simulaotr
       ``'INI'`` only uses a threshold, reset and membrane time constant.
    3. Simulation parameters ``simparams``, specifying e.g. length and
       time resolution of the simulation run.

Parameters
..........

Global Parameters
*****************

dataset : string
    The dataset to use for testing the network. Possible values:

    - 'mnist'
    - 'cifar10'

architecture : string
    Specifies the network architecture. Possible values:

    - 'mlp' for Multi-Layer Perceptron
    - 'cnn' for Convolutional Neural Network

model_lib : string
    The neural network library used to build the ANN, e.g.

    - 'keras'
    - 'lasagne'
    - 'caffe'

path : string, optional
    Location of the ANN model to load. If not specified, the toolbox will use
    as destination for all files it needs to load and save:
    ``<repo-root>/data/<dataset>/<architecture>/<filename>/<simulator>/``.
    For instance, if the repo is in ``~/snntoolbox/`` and we specify as
    filename of the ANN model to load: ``globalparams['filename'] = '98.29'``
    and use default parameters otherwise, the toolbox will perform all
    io-operations in ``~/snntoolbox/data/mnist/mlp/98.29/INI/``.
filename : string
    Name of the json file containing the model architecture and weights.
batch_size : int, optional
    Size of batch to test.
debug : boolean, optional
    If true, the input data is reduced to one batch_size to reduce
    computation time.
evaluateANN : boolean, optional
    Whether or not to test the pretrained ANN before conversion to spiking net.
sim_only : boolean, optional
    If true, skip conversion step and try to load SNN from ``/<path>/``.
normalize : boolean, optional
    If true, the network will first perform a simulation using the original
    weights, and then normalize the weights and run again both ANN and SNN
    with modified weights to evaluate the impact of weight normalization
    on accuracy.
overwrite : boolean, optional
    If false, the save methods will ask for permission to overwrite files
    before writing weights, activations, models etc. to disk.
verbose : int, optional
    :0: No intermediate results or status reports.
    :1: Print progress of simulation and intermediate results.
    :2: After each batch, plot guessed classes per sample and show an
        input image. At the end of the simulation, plot the number of spikes
        for each sample.
    :3: Record, plot and return the membrane potential of all
        layers for the last test sample. Very time consuming.

Cell Parameters
***************

v_thresh : float, optional
    Threshold in mV defining the voltage at which a spike is fired.
v_reset : float, optional
    Reset potential in mV of the neurons after spiking.
v_rest : float, optional
    Resting membrane potential in mV.
e_rev_E : float, optional
    Reversal potential for excitatory input in mV.
e_rev_I : float, optional
    Reversal potential for inhibitory input in mV.
i_offset : float, optional
    Offset current in nA.
cm : float, optional
    Membrane capacitance in nF.
tau_m : float, optional
    Membrane time constant in milliseconds.
tau_refrac : float, optional
    Duration of refractory period in milliseconds of the neurons after spiking.
tau_syn_E : float, optional
    Decay time of the excitatory synaptic conductance in milliseconds.
tau_syn_I : float, optional
    Decay time of the inhibitory synaptic conductance in milliseconds.

Simulation Parameters
*********************

duration : float, optional
    Runtime of simulation of one input in milliseconds.
dt : float, optional
    Time resolution of spikes in milliseconds.
delay : float, optional
    Delay in milliseconds. Must be equal to or greater than the resolution.
max_f : float, optional
    Spike rate in Hz for a fully-on pixel.
num_to_test : int, optional
    How many samples to test.

Default values
..............

::

    globalparams = {'dataset': 'mnist',
                    'architecture': 'mlp',
                    'model_lib': 'keras',
                    'path': '',
                    'filename': '',
                    'batch_size': 100,
                    'debug': False,
                    'evaluateANN': True,
                    'normalize': True,
                    'overwrite': True,
                    'sim_only': False,
                    'verbose': 2}
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
                  'tau_syn_I': 0.01}
    simparams = {'duration': 200,
                 'dt': 1,
                 'delay': 1,
                 'max_f': 1000,
                 'num_to_test': 10}


Switching Simulators
....................

When running the SNN toolbox for the first time, it will create a configuration
file in your home directory:

``~/.snntoolbox/snntoolbox.json``

(You can of course create it yourself.)

It contains a dictionary of configuration options:

``{'simulator': 'INI'}``

Change the ``simulator`` key to any simulator you installed and which supports
pyNN. The modified settings will be loaded the next time you use any part of
the toolbox.

Simulators currently supported by pyNN include

    - ``'nest'``
    - ``'brian'``
    - ``'Neuron'``.

In addition, the toolbox includes as default our own simulator ``'INI'``.

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

# List supported datasets, model types and model libraries.
datasets = {'mnist', 'cifar10', 'caltech101'}
datasetsGray = {'mnist'}
datasetsRGB = {'cifar10', 'caltech101'}
architectures = {'mlp', 'cnn'}
model_libs = {'keras', 'lasagne'}

# Default parameters:
globalparams = {'dataset': 'mnist',
                'architecture': 'mlp',
                'model_lib': 'keras',
                'path': '',
                'filename': '',
                'batch_size': 100,
                'debug': False,
                'evaluateANN': True,
                'normalize': True,
                'overwrite': True,
                'sim_only': False,
                'verbose': 2}
cellparams = {'v_thresh': 1,
              'tau_refrac': 0}  # No refractory period
simparams = {'duration': 200,
             'dt': 1,
             'max_f': 1000}

# pyNN specific parameters will be added to parameters above when initializing
# the toolbox.
cellparams_pyNN = {'v_reset': 0,
                   'v_rest': 0,  # Initial neuron potential
                   'e_rev_E': 10,  # Pull v towards +10 mV during spike
                   'e_rev_I': -10,  # Pull v towards -10 mV during spike
                   'i_offset': 0,  # No offset current
                   'cm': 0.09,  # Fast integration. Small cm slows down sim.
                   'tau_m': 1000,  # No leakage
                   'tau_syn_E': 0.01,  # Excitatory synaptic cond decays fast
                   'tau_syn_I': 0.01}  # Inhibitory synaptic cond decays fast
simparams_pyNN = {'delay': 1,  # duration must be longer than delay*num_layers
                  'num_to_test': 10}  # Constraint: delay >= dt

# Layers followed by an Activation layer
activation_layers = {'Dense', 'Convolution2D'}

bn_layers = {'Dense', 'Convolution2D'}


def update_setup(global_params={}, cell_params={}, sim_params={}):
    """
    Update parameters

    Check that parameter choices are valid and update the default parameters
    with the user-specified values.
    """

    import os
    import snntoolbox

    if 'dataset' in global_params:
        assert global_params['dataset'] in datasets, \
            "Dataset '{}' not known. Supported datasets: {}".format(
                global_params['dataset'], datasets)
    if 'architecture' in global_params:
        assert global_params['architecture'] in architectures, \
            "Network architecture '{}' not understood. Supported architectures:\
                {}".format(global_params['architecture'], architectures)
    if 'model_lib' in global_params:
        assert global_params['model_lib'] in model_libs, \
            "Input model library '{}' ".format(global_params['model_lib']) + \
            "not supported yet. Possible values: {}".format(model_libs)
    assert 'filename' in global_params, \
        "Filename of stored model not specified."
    if 'path' not in global_params or global_params['path'] == '':
        global_params['path'] = os.path.join(snntoolbox._dir, 'data',
                                             global_params['dataset'],
                                             global_params['architecture'],
                                             global_params['filename'],
                                             snntoolbox._SIMULATOR)
    else:
        if not os.path.exists(global_params['path']):
            os.makedirs(global_params['path'])
    if 'simulator' in global_params:
        assert global_params['simulator'] in snntoolbox.simulators, \
            "Simulator '{}' not supported. ".format(
            global_params['simulator']) + "Choose from {}".format(
                                                        snntoolbox.simulators)
        snntoolbox.sim, snntoolbox.custom_layers = \
            snntoolbox.initialize_simulator(global_params['simulator'])
    if snntoolbox._SIMULATOR == 'brian2' and 'sim_only' in global_params and \
            global_params['sim_only'] or 'sim_only' not in global_params and \
            globalparams['sim_only']:
        print('\n')
        print("SNN toolbox Warning: When using Brian 2 simulator, you need " +
              "to convert the network each time you start a new session. " +
              "(No saving/reloading methods implemented.) Setting " +
              "globalparams['sim_only'] = False.\n")
        global_params['sim_only'] = False
#    if snntoolbox._SIMULATOR == 'INI' and 'sim_only' in global_params and \
#            global_params['sim_only'] or 'sim_only' not in global_params and \
#            globalparams['sim_only']:
#        print('\n')
#        print("SNN toolbox Warning: When using INI simulator, you need " +
#              "to convert the network each time you start a new session. " +
#              "(No saving/reloading methods implemented yet.) Setting " +
#              "globalparams['sim_only'] = False.\n")
#        global_params['sim_only'] = False

    # If there are any parameters specified, merge with default parameters.
    globalparams.update(global_params)
    cellparams.update(cell_params)
    simparams.update(sim_params)
