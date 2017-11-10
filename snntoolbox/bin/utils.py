# -*- coding: utf-8 -*-
"""
This module bundles all the tools of the SNN conversion toolbox.

Important functions:

.. autosummary::
    :nosignatures:

    test_full
    update_setup

@author: rbodo
"""

from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import os
from importlib import import_module

from future import standard_library

standard_library.install_aliases()


def test_full(config, queue=None):
    """Convert an analog network to a spiking network and simulate it.

    Complete pipeline of
        1. loading and testing a pretrained ANN,
        2. normalizing parameters
        3. converting it to SNN,
        4. running it on a simulator,
        5. given a specified hyperparameter range ``params``,
           repeat simulations with modified parameters.

    Parameters
    ----------

    config: configparser.ConfigParser
        ConfigParser containing the user settings.

    queue: Optional[Queue.Queue]
        Results are added to the queue to be displayed in the GUI.

    Returns
    -------

    results: list
        List of the accuracies obtained after simulating with each parameter
        value in config.get('parameter_sweep', 'param_values').
    """

    from snntoolbox.datasets.utils import get_dataset
    from snntoolbox.conversion.utils import normalize_parameters

    num_to_test = config.getint('simulation', 'num_to_test')

    # Instantiate an empty spiking network
    target_sim = import_module('snntoolbox.simulation.target_simulators.' +
                               config.get('simulation', 'simulator') +
                               '_target_sim')
    spiking_model = target_sim.SNN(config, queue)

    # ____________________________ LOAD DATASET ______________________________ #

    normset, testset = get_dataset(config)

    if config.getboolean('tools', 'convert') and not is_stop(queue):

        # ___________________________ LOAD MODEL _____________________________ #

        model_lib = import_module('snntoolbox.parsing.model_libs.' +
                                  config.get('input', 'model_lib') +
                                  '_input_lib')
        input_model = model_lib.load(config.get('paths', 'path_wd'),
                                     config.get('paths', 'filename_ann'))

        # Evaluate input model.
        if config.getboolean('tools', 'evaluate_ann') and not is_stop(queue):
            print("Evaluating input model on {} samples...".format(num_to_test))
            model_lib.evaluate(input_model['val_fn'],
                               config.getint('simulation', 'batch_size'),
                               num_to_test, **testset)

        # _____________________________ PARSE ________________________________ #

        print("Parsing input model...")
        model_parser = model_lib.ModelParser(input_model['model'], config)
        model_parser.parse()
        parsed_model = model_parser.build_parsed_model()

        # ____________________________ NORMALIZE _____________________________ #

        if config.getboolean('tools', 'normalize') and not is_stop(queue):
            normalize_parameters(parsed_model, config, **normset)

        # Evaluate parsed model.
        if config.getboolean('tools', 'evaluate_ann') and not is_stop(queue):
            print("Evaluating parsed model on {} samples...".format(
                num_to_test))
            model_parser.evaluate_parsed(config.getint(
                'simulation', 'batch_size'), num_to_test, **testset)

        # Write parsed model to disk
        parsed_model.save(
            os.path.join(config.get('paths', 'path_wd'),
                         config.get('paths', 'filename_parsed_model') + '.h5'))

        # ____________________________ CONVERT _______________________________ #

        spiking_model.build(parsed_model)

        # Export network in a format specific to the simulator with which it
        # will be tested later.
        spiking_model.save(config.get('paths', 'path_wd'),
                           config.get('paths', 'filename_snn'))

    # _______________________________ SIMULATE _______________________________ #

    if config.getboolean('tools', 'simulate') and not is_stop(queue):

        # Decorate the 'run' function of the spiking model with a parameter
        # sweep function.
        @run_parameter_sweep(config, queue)
        def run(snn, **test_set):
            return snn.run(**test_set)

        # Simulate network
        results = run(spiking_model, **testset)

        # Clean up
        spiking_model.end_sim()

        # Add results to queue to be displayed in GUI.
        if queue:
            queue.put(results)

        return results


def is_stop(queue):
    """Determine if the user pressed 'stop' in the GUI.

    Parameters
    ----------

    queue: Queue.Queue
        Event queue.

    Returns
    -------

    : bool
        ``True`` if user pressed 'stop' in GUI, ``False`` otherwise.
    """

    if not queue:
        return False
    if queue.empty():
        return False
    elif queue.get_nowait() == 'stop':
        print("Skipped step after user interrupt")
        queue.put('stop')
        return True


def run_parameter_sweep(config, queue):
    """
    Decorator to perform a parameter sweep using the ``run_single`` function.
    Need an aditional wrapping layer to be able to pass decorator arguments.
    """

    def decorator(run_single):

        from functools import wraps

        @wraps(run_single)
        def wrapper(snn, **testset):

            results = []
            param_values = eval(config.get('parameter_sweep', 'param_values'))
            param_name = config.get('parameter_sweep', 'param_name')
            param_logscale = config.getboolean('parameter_sweep',
                                               'param_logscale')
            if len(param_values) > 1:
                print("Testing SNN for parameter values {} = ".format(
                    param_name))
                print(['{:.2f}'.format(i) for i in param_values])
                print('\n')

            # Loop over parameter to sweep
            for p in param_values:
                if is_stop(queue):
                    break

                # Display current parameter value
                config.set('cell', param_name, str(p))
                if len(param_values) > 1:
                    print("\nCurrent value of parameter to sweep: " +
                          "{} = {:.2f}\n".format(param_name, p))

                results.append(run_single(snn, **testset))

            # Plot and return results of parameter sweep.
            try:
                from snntoolbox.simulation.plotting import plot_param_sweep
            except ImportError:
                plot_param_sweep = None
            if plot_param_sweep is not None:
                plot_param_sweep(
                    results, config.getint('simulation', 'num_to_test'),
                    param_values, param_name, param_logscale)

            return results
        return wrapper
    return decorator


def load_config(filepath):
    """
    Load a config file from ``filepath``.
    """

    try:
        import configparser
    except ImportError:
        # noinspection PyPep8Naming
        import ConfigParser as configparser
        # noinspection PyUnboundLocalVariable
        configparser = configparser

    assert os.path.isfile(filepath), \
        "Configuration file not found at {}.".format(filepath)

    config = configparser.ConfigParser()
    config.read(filepath)

    return config


def update_setup(config_filepath):
    """Update default settings with user settings and check they are valid.

    Load settings from configuration file at ``config_filepath``, and check that
    parameter choices are valid. Non-specified settings are filled in with
    defaults.
    """

    from textwrap import dedent

    # Load defaults.
    config = load_config(os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', 'config_defaults')))

    # Overwrite with user settings.
    config.read(config_filepath)

    # Name of input file must be given.
    filename_ann = config.get('paths', 'filename_ann')
    assert filename_ann != '', "Filename of input model not specified."

    # Check that simulator choice is valid.
    simulator = config.get('simulation', 'simulator')
    simulators = eval(config.get('restrictions', 'simulators'))
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
    if config.get('paths', 'path_wd') == '':
        config.set('paths', 'path_wd', os.path.dirname(config_filepath))

    # Check specified working directory exists.
    path_wd = config.get('paths', 'path_wd')
    assert os.path.exists(path_wd), \
        "Working directory {} does not exist.".format(path_wd)

    # Check that choice of input model library is valid.
    model_lib = config.get('input', 'model_lib')
    model_libs = eval(config.get('restrictions', 'model_libs'))
    assert model_lib in model_libs, "ERROR: Input model library '{}' ".format(
        model_lib) + "not supported yet. Possible values: {}".format(model_libs)

    # Check input model is found and has the right format for the specified
    # model library.
    if model_lib == 'caffe':
        caffemodel_filepath = os.path.join(path_wd,
                                           filename_ann + '.caffemodel')
        caffemodel_h5_filepath = os.path.join(path_wd,
                                              filename_ann + '.caffemodel.h5')
        assert os.path.isfile(caffemodel_filepath) or os.path.isfile(
            caffemodel_h5_filepath), "File {} or {} not found.".format(
            caffemodel_filepath, caffemodel_h5_filepath)
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
            from snntoolbox.parsing.utils import get_custom_activations_dict
            try:
                keras.models.load_model(h5_filepath,
                                        get_custom_activations_dict())
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
    if config.get('paths', 'dataset_path') == '':
        config.set('paths', 'dataset_path', os.path.dirname(__file__))

    # Check that the data set path is valid.
    dataset_path = os.path.abspath(config.get('paths', 'dataset_path'))
    config.set('paths', 'dataset_path', dataset_path)
    assert os.path.exists(dataset_path), "Path to data set does not exist: " \
                                         "{}".format(dataset_path)

    # Check that data set path contains the data in the specified format.
    assert os.listdir(dataset_path), "Data set directory is empty."
    normalize = config.getboolean('tools', 'normalize')
    dataset_format = config.get('input', 'dataset_format')
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

    sample_idxs_to_test = eval(config.get('simulation', 'sample_idxs_to_test'))
    num_to_test = config.getint('simulation', 'num_to_test')
    if not sample_idxs_to_test == []:
        if len(sample_idxs_to_test) != num_to_test:
            print(dedent("""
            SNN toolbox warning: Settings mismatch. Adjusting 'num_to_test' to 
            equal the number of 'sample_idxs_to_test'."""))
            config.set('simulation', 'num_to_test',
                       str(len(sample_idxs_to_test)))

    # Create log directory if it does not exist.
    if config.get('paths', 'log_dir_of_current_run') == '':
        config.set('paths', 'log_dir_of_current_run', os.path.join(
            path_wd, 'log', 'gui', config.get('paths', 'runlabel')))
    log_dir_of_current_run = config.get('paths', 'log_dir_of_current_run')
    if not os.path.isdir(log_dir_of_current_run):
        os.makedirs(log_dir_of_current_run)

    # Specify filenames for models at different stages of the conversion.
    if config.get('paths', 'filename_parsed_model') == '':
        config.set('paths', 'filename_parsed_model', filename_ann + '_parsed')
    if config.get('paths', 'filename_snn') == '':
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

    plot_var = get_plot_keys(config)
    plot_vars = eval(config.get('restrictions', 'plot_vars'))
    assert all([v in plot_vars for v in plot_var]), \
        "Plot variable(s) {} not understood.".format(
            [v for v in plot_var if v not in plot_vars])
    if 'all' in plot_var:
        plot_vars_all = plot_vars.copy()
        plot_vars_all.remove('all')
        config.set('output', 'plot_vars', str(plot_vars_all))

    log_var = get_log_keys(config)
    log_vars = eval(config.get('restrictions', 'log_vars'))
    assert all([v in log_vars for v in log_var]), \
        "Log variable(s) {} not understood.".format(
            [v for v in log_var if v not in log_vars])
    if 'all' in log_var:
        log_vars_all = log_vars.copy()
        log_vars_all.remove('all')
        config.set('output', 'log_vars', str(log_vars_all))

    # Change matplotlib plot properties, e.g. label font size
    try:
        import matplotlib
    except ImportError:
        matplotlib = None
        if len(plot_vars) > 0:
            import warnings
            warnings.warn("Package 'matplotlib' not installed; disabling "
                          "plotting. Run 'pip install matplotlib' to enable "
                          "plotting.", ImportWarning)
            config.set('output', 'plot_vars', str({}))
    if matplotlib is not None:
        matplotlib.rcParams.update(eval(config.get('output', 'plotproperties')))

    # Check settings for parameter sweep
    param_name = config.get('parameter_sweep', 'param_name')
    try:
        config.get('cell', param_name)
    except KeyError:
        print("Unkown parameter name {} to sweep.".format(param_name))
        raise RuntimeError
    if not eval(config.get('parameter_sweep', 'param_values')):
        config.set('parameter_sweep', 'param_values',
                   str([eval(config.get('cell', param_name))]))

    if config.getboolean('conversion', 'use_isi_code'):
        config.set('cell', 'tau_refrac',
                   str(config.getint('simulation', 'duration')))
        config.set('conversion', 'softmax_to_relu', 'True')

    with open(os.path.join(log_dir_of_current_run, '.config'), str('w')) as f:
        config.write(f)

    return config


def initialize_simulator(config):
    """Import a module that contains utility functions of spiking simulator."""
    from importlib import import_module

    sim = None
    simulator = config.get('simulation', 'simulator')
    if simulator in eval(config.get('restrictions', 'simulators_pyNN')):
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
        sim.setup(timestep=config.getfloat('simulation', 'dt'))
    elif simulator == 'brian2':
        sim = import_module('brian2')
    elif simulator == 'INI':
        sim = import_module('snntoolbox.simulation.backends.inisim.inisim')
    elif simulator == 'MegaSim':
        sim = import_module('snntoolbox.simulation.backends.megasim.megasim')
    elif simulator == 'INIed':
        sim = import_module('snntoolbox.simulation.backends.inisim.inied')
    assert sim, "Simulator {} could not be initialized.".format(simulator)
    print("Initialized {} simulator.\n".format(simulator))
    return sim


def get_log_keys(config):
    return set(eval(config.get('output', 'log_vars')))


def get_plot_keys(config):
    return set(eval(config.get('output', 'plot_vars')))
