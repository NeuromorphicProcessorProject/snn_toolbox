# coding=utf-8

"""Configure SNN toolbox and check that user specified settings are valid."""

import os

# List supported model libraries, simulators, etc.
model_libs = {'keras', 'lasagne', 'caffe'}
maxpool_types = {'fir_max', 'exp_max', 'avg_max'}
simulators_pyNN = {'nest', 'brian', 'Neuron'}
simulators_other = {'INI', 'brian2', 'MegaSim'}
simulators = simulators_pyNN.copy()
simulators |= simulators_other

# Possible variables to monitor and save / plot:
log_vars = {'activations_n_b_l', 'spiketrains_n_b_l_t', 'input_b_l_t',
            'mem_n_b_l_t', 'operations_b_t', 'all'}
plot_vars = {'activations', 'spiketrains', 'spikecounts', 'spikerates',
             'input_image', 'error_t', 'confusion_matrix', 'correlation',
             'hist_spikerates_activations', 'normalization_activations',
             'operations', 'all'}

# Layers that can be implemented by our spiking neuron simulators
spiking_layers = {'Dense', 'Conv2D', 'MaxPooling2D', 'AveragePooling2D'}

snn_layers = spiking_layers | {'Flatten', 'Concatenate'}

# pyNN specific parameters.
pyNN_keys = {
    'v_reset',
    'v_rest',  # Initial neuron potential
    'e_rev_E',  # Pull v towards +10 mV during spike
    'e_rev_I',  # Pull v towards -10 mV during spike
    'i_offset',  # No offset current
    'cm',  # Fast integration. Small cm slows down sim.
    'tau_m',  # No leakage
    'tau_syn_E',  # Excitatory synaptic cond decays fast
    'tau_syn_I',  # Inhibitory synaptic cond decays fast
    'delay'}  # Constraint: delay >= dt


def load_config(filepath):
    """
    Load a config file from ``filepath``.
    """

    try:
        import configparser
    except ImportError:
        # noinspection PyPep8Naming
        import ConfigParser as configparser

    assert os.path.isfile(filepath), \
        "Configuration file not found at {}.".format(filepath)

    config = configparser.ConfigParser()
    config.read(filepath)

    return config


def update_setup(config_filepath):
    """Update parameters.

    Load settings from configuration file at ``config_filepath``, and check that
    parameter choices are valid. Non-specified settings are filled in with
    defaults.
    """

    import matplotlib as mpl
    from textwrap import dedent

    # Load defaults.
    config = load_config(os.path.join(os.path.dirname(__file__),
                                      'config_defaults'))

    # Overwrite with user settings.
    config.read(config_filepath)

    # Name of input file must be given.
    filename_ann = config['paths']['filename_ann']
    assert filename_ann != '', "Filename of input model not specified."

    # Check that simulator choice is valid.
    simulator = config['simulation']['simulator']
    assert simulator in simulators, \
        "Simulator '{}' not supported. Choose from {}".format(simulator,
                                                              simulators)

    # Warn user that it is not possible to use Brian2 simulator by loading a
    # pre-converted network from disk.
    if simulator == 'brian2' and not config.getboolean('tools', 'convert'):
        print(dedent("""\ \n
            SNN toolbox Warning: When using Brian 2 simulator, you need to
            convert the network each time you start a new session. (No
            saving/reloading methods implemented.) Setting convert = True.
            \n"""))
        config.set('tools', 'convert', str(True))

    # Set default path if user did not specify it.
    if config['paths']['path_wd'] == '':
        config.set('paths', 'path_wd', os.path.dirname(config_filepath))

    # Check specified working directory exists.
    path_wd = config['paths']['path_wd']
    assert os.path.exists(path_wd), \
        "Working directory {} does not exist.".format(path_wd)

    # Check that choice of input model library is valid.
    model_lib = config['input']['model_lib']
    assert model_lib in model_libs, "ERROR: Input model library '{}' ".format(
        model_lib) + "not supported yet. Possible values: {}".format(model_libs)

    # Check input model is found and has the right format for the specified
    # model library.
    if model_lib == 'caffe':
        caffemodel_filepath = os.path.join(path_wd,
                                           filename_ann + '.caffemodel')
        assert os.path.isfile(caffemodel_filepath), \
            "File {} not found.".format(caffemodel_filepath)
        prototxt_filepath = os.path.join(path_wd, filename_ann + '.prototxt')
        assert os.path.isfile(prototxt_filepath), \
            "File {} not found.".format(prototxt_filepath)
    elif model_lib == 'keras':
        h5_filepath = os.path.join(path_wd, filename_ann + '.h5')
        assert os.path.isfile(h5_filepath), \
            "File {} not found.".format(h5_filepath)
        json_file = filename_ann + '.json'
        if not os.path.isfile(os.path.join(path_wd, json_file)):
            import keras
            # from snntoolbox.core.util import binary_sigmoid, binary_tanh
            custom_objects = None  # {'binary_sigmoid': binary_sigmoid}
            try:
                keras.models.load_model(h5_filepath, custom_objects)
            except:
                raise AssertionError(
                    "Input model could not be loaded. This is likely due to a "
                    "Keras version backwards-incompability. For instance, you "
                    "might have provided an h5 file with weights, but without "
                    "network configuration. In earlier versions of Keras, this "
                    "is contained in a json file.")
    elif model_lib == 'lasagne':
        h5_filepath = os.path.join(path_wd, filename_ann + '.h5')
        pkl_filepath = os.path.join(path_wd, filename_ann + '.pkl')
        assert os.path.isfile(h5_filepath) or os.path.isfile(pkl_filepath), \
            "File {} not found.".format('.h5 or .pkl')
        py_filepath = os.path.join(path_wd, filename_ann + '.py')
        assert os.path.isfile(py_filepath), \
            "File {} not found.".format(py_filepath)
    else:
        print("For the specified input model library {}, ".format(model_lib) +
              "no test is implemented to check if input model files exist in "
              "the specified working directory!")

    # Set default path if user did not specify it.
    if config['paths']['dataset_path'] == '':
        config.set('paths', 'dataset_path', os.path.dirname(__file__))

    # Check that the data set path is valid.
    dataset_path = os.path.abspath(config['paths']['dataset_path'])
    config.set('paths', 'dataset_path', dataset_path)
    assert os.path.exists(dataset_path), "Path to data set does not exist: " \
                                         "{}".format(dataset_path)

    # Check that data set path contains the data in the specified format.
    assert os.listdir(dataset_path), "Data set directory is empty."
    normalize = config.getboolean('tools', 'normalize')
    dataset_format = config['input']['dataset_format']
    if dataset_format == 'npz' and normalize and not os.path.exists(
            os.path.join(dataset_path, 'x_norm.npz')):
        raise RuntimeWarning(
            "No data set file 'x_norm.npz' found in specified data set path " +
            "{}. Add it, or disable normalization.".format(dataset_path))
    if dataset_format == 'npz' and not (os.path.exists(os.path.join(
            dataset_path, 'x_test.npz')) and os.path.exists(os.path.join(
            dataset_path, 'y_test.npz'))):
        raise RuntimeWarning(
            "Data set file 'x_test.npz' or 'y_test.npz' was not found in "
            "specified data set path {}.".format(dataset_path))

    sample_idxs_to_test = eval(config['simulation']['sample_idxs_to_test'])
    num_to_test = config.getint('simulation', 'num_to_test')
    if not sample_idxs_to_test == []:
        if len(sample_idxs_to_test) != num_to_test:
            print(dedent("""
            SNN toolbox warning: Settings mismatch. Adjusting 'num_to_test' to 
            equal the number of 'sample_idxs_to_test'."""))
            config.set('simulation', 'num_to_test',
                       str(len(sample_idxs_to_test)))

    # Create log directory if it does not exist.
    if config['paths']['log_dir_of_current_run'] == '':
        config.set('paths', 'log_dir_of_current_run', os.path.join(
            path_wd, 'log', 'gui', config['paths']['runlabel']))
    log_dir_of_current_run = config['paths']['log_dir_of_current_run']
    if not os.path.isdir(log_dir_of_current_run):
        os.makedirs(log_dir_of_current_run)

    # Specify filenames for models at different stages of the conversion.
    if config['paths']['filename_parsed_model'] == '':
        config.set('paths', 'filename_parsed_model', filename_ann + '_parsed')
    if config['paths']['filename_snn'] == '':
        config.set('paths', 'filename_snn', '{}_{}'.format(filename_ann,
                                                           simulator))

    if simulator != 'INI' and not config.getboolean('input', 'poisson_input'):
        config.set('input', 'poisson_input', str(True))
        print(dedent("""\
            SNN toolbox Warning: Currently, turning off Poisson input is
            only possible in INI simulator. Falling back on Poisson input."""))

    # Make sure the number of samples to test is not lower than the batch size.
    batch_size = config.getint('simulation', 'batch_size')
    if config.getint('simulation', 'num_to_test') < batch_size:
        print(dedent("""\
            SNN toolbox Warning: 'num_to_test' set lower than 'batch_size'.
            In simulators that test samples batch-wise (e.g. INIsim), this
            can lead to undesired behavior. Setting 'num_to_test' equal to
            'batch_size'."""))
        config.set('simulation', 'num_to_test', str(batch_size))

    from snntoolbox.core.util import get_plot_keys, get_log_keys

    plot_var = get_plot_keys(config)
    assert all([v in plot_vars for v in plot_var]), \
        "Plot variable(s) {} not understood.".format(
            [v for v in plot_var if v not in plot_vars])

    log_var = get_log_keys(config)
    assert all([v in log_vars for v in log_var]), \
        "Log variable(s) {} not understood.".format(
            [v for v in log_var if v not in log_vars])

    if 'all' in plot_var:
        config.set('output', 'plot_vars', str(plot_vars))

    if 'all' in log_var:
        config.set('output', 'log_vars', str(log_vars))

    # Change matplotlib plot properties, e.g. label font size
    mpl.rcParams.update(eval(config.get('output', 'plotproperties')))

    # Check settings for parameter sweep
    param_name = config['parameter_sweep']['param_name']
    assert param_name in config['cell'], \
        "Unkown parameter name {} to sweep.".format(param_name)
    if not eval(config['parameter_sweep']['param_values']):
        config.set('parameter_sweep', 'param_values',
                   str([eval(config['cell'][param_name])]))

    with open(os.path.join(log_dir_of_current_run, '.config'), 'w') as f:
        config.write(f)

    return config


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
