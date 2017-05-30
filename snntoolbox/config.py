# coding=utf-8

"""Configure SNN toolbox and check that user specified settings are valid."""

import os
import matplotlib as mpl
from textwrap import dedent
from collections import OrderedDict

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
maxpool_types = {'fir_max', 'exp_max', 'avg_max'}
custom_activations = {'binary_tanh', 'binary_sigmoid'}
simulators_pyNN = {'nest', 'brian', 'Neuron'}
simulators_other = {'INI', 'brian2', 'MegaSim'}
simulators = simulators_pyNN.copy()
simulators |= simulators_other

# Default parameters:
path_default = os.path.join(os.path.dirname(__file__), 'default_settings.txt')
settings = OrderedDict(eval(open(path_default, 'r').read()))

# Possible variables to monitor and save / plot:
log_vars = {'activations_n_b_l', 'spiketrains_n_b_l_t', 'input_b_l_t',
            'mem_n_b_l_t', 'operations_b_t', 'all'}
plot_vars = {'activations', 'spiketrains', 'spikecounts', 'spikerates',
             'input_image', 'error_t', 'confusion_matrix', 'correlation',
             'hist_spikerates_activations', 'normalization_activations',
             'operations', 'all'}

# pyNN specific parameters.
pyNN_keys = [
    'v_reset',
    'v_rest',  # Initial neuron potential
    'e_rev_E',  # Pull v towards +10 mV during spike
    'e_rev_I',  # Pull v towards -10 mV during spike
    'i_offset',  # No offset current
    'cm',  # Fast integration. Small cm slows down sim.
    'tau_m',  # No leakage
    'tau_syn_E',  # Excitatory synaptic cond decays fast
    'tau_syn_I',  # Inhibitory synaptic cond decays fast
    'delay']  # Constraint: delay >= dt

pyNN_settings = {}
for key in pyNN_keys:
    pyNN_settings[key] = settings[key]

# Layers that can be implemented by our spiking neuron simulators
spiking_layers = ['Dense', 'Conv2D', 'MaxPooling2D', 'AveragePooling2D',
                  'Flatten', 'Concatenate']


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
            # TODO: enable loading custom activation functions
            # from snntoolbox.core.util import binary_sigmoid, binary_tanh
            # import keras.activations
            # keras.activations.binary_sigmoid = binary_sigmoid
            custom_objects = None  # {'binary_sigmoid': binary_sigmoid}
            try:
                keras.models.load_model(
                    h5_filepath, custom_objects)
            except:
                raise AssertionError(
                    "Input model could not be loaded. This is likely due to a "
                    "Keras version backwards-incompability. For instance, you "
                    "might have provided an h5 file with weights, but without "
                    "network configuration. In earlier versions of Keras, this "
                    "is contained in a json file.")
    elif settings['model_lib'] == 'lasagne':
        h5_file = settings['filename_ann'] + '.h5'
        pkl_file = settings['filename_ann'] + '.pkl'
        assert os.path.isfile(os.path.join(settings['path_wd'], h5_file)) or \
            os.path.isfile(os.path.join(settings['path_wd'], pkl_file)), \
            file_not_found_msg('.h5 or .pkl')
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
        assert os.listdir(settings['dataset_path']), "Data set directory is " \
                                                     "empty."
        # Transform str to dict
        settings['datagen_kwargs'] = eval(settings['datagen_kwargs'])
        settings['dataflow_kwargs'] = eval(settings['dataflow_kwargs'])

        # Get class labels
        if 'class_idx_path' in settings['dataflow_kwargs']:
            import json
            class_idx = json.load(open(
                settings['dataflow_kwargs'].pop('class_idx_path'), 'r'))
            classes = [class_idx[str(idx)][0] for idx in range(len(class_idx))]
            settings['dataflow_kwargs']['classes'] = classes

        # Get proprocessing function
        if 'preprocessing_function' in settings['datagen_kwargs']:
            from snntoolbox.model_libs.common import import_script
            helpers = import_script(
                '.', settings['datagen_kwargs']['preprocessing_function'])
            settings['datagen_kwargs']['preprocessing_function'] = \
                helpers.preprocessing_function
    if settings['dataset_format'] == 'aedat':
        assert os.listdir(settings['dataset_path']), "Data set directory is " \
                                                     "empty."

    # Convert string containing sample indices to list of indices.
    assert isinstance(settings['samples_to_test'], str), "The parameter " + \
        "'samples_to_test' must be of type 'string'. Given: {}".format(
        settings['samples_to_test'].__class__)
    settings['sample_indices_to_test'] = [
        int(i) for i in settings['samples_to_test'].split() if i.isnumeric()]
    if not settings['sample_indices_to_test'] == []:
        if len(settings['sample_indices_to_test']) != settings['num_to_test']:
            print(dedent("""
            SNN toolbox Warning: Settings mismatch. Adjusting 'num_to_test' to
            equal the number of 'samples_to_test'."""))
            settings['num_to_test'] = len(settings['sample_indices_to_test'])
        if settings['dataset_format'] == 'jpg':
            assert "'shuffle': False" in settings['dataflow_kwargs'], dedent("""
                SNN toolbox Warning: You gave a list of specific samples to be
                tested. For this to work in combination with a  
                DataImageGenerator, you should set shuffling=False in 
                settings['dataflow_kwargs'].""")

    # Create log directory if it does not exist.
    if 'log_dir_of_current_run' not in s:
        settings['log_dir_of_current_run'] = os.path.join(
            s['path_wd'], 'log', 'gui', settings['runlabel'])
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

    if not settings['convert']:
        print("SNN toolbox Warning: You have restored a previously converted "
              "SNN from disk. If this net uses an activation function unknown "
              "to Keras, this custom function will not have been saved and "
              "reloaded, but replaced by default 'linear'. Convert from "
              "scratch before simulating to use custom function.")

    assert all([v in plot_vars for v in settings['plot_vars']]), \
        "Plot variable(s) {} not understood.".format(
            [v for v in settings['plot_vars'] if v not in plot_vars])

    assert all([v in log_vars for v in settings['log_vars']]), \
        "Log variable(s) {} not understood.".format(
            [v for v in settings['log_vars'] if v not in log_vars])

    if 'all' in settings['log_vars']:
        settings['log_vars'] = log_vars
        settings['log_vars'].remove('all')

    if 'all' in settings['plot_vars']:
        settings['plot_vars'] = plot_vars
        settings['plot_vars'].remove('all')

    if settings['custom_activation'] is not None:
        assert settings['custom_activation'] in custom_activations, \
            "Custom activation {} not understood. Choose from {}.".format(
                settings['custom_activation'], custom_activations)

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
    elif simulator == 'INIed':
        sim = import_module('snntoolbox.core.inied')
    assert sim, "Simulator {} could not be initialized.".format(simulator)
    print("Initialized {} simulator.\n".format(simulator))
    return sim
