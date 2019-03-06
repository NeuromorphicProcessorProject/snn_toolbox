from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import os
from importlib import import_module
from textwrap import dedent

from future import standard_library

import snntoolbox
from snntoolbox.bin.utils import config_string_to_set_of_strings, load_config, \
    get_plot_keys, get_log_keys, is_stop
from snntoolbox.datasets.utils import get_dataset
from snntoolbox.conversion.utils import normalize_parameters
from snntoolbox.simulation.target_simulators.spiNNaker_target_sim import SNN

standard_library.install_aliases()

# CHANGE THIS TO POINT TO YOUR CONFIG FILE:
config_filepath = '/mnt/2646BAF446BAC3B9/Data/snn_conversion/mnist/cnn/lenet5' \
                  '/keras/pyNN/channels_last/log/gui/01/config'

"""++++++++++++++++++++++++ CHECK SETTINGS ++++++++++++++++++++++++++++++++++"""

# Load defaults.
config = load_config(os.path.abspath(os.path.join(
    os.path.dirname(snntoolbox.__file__), 'config_defaults')))

# Overwrite with user settings.
config.read(config_filepath)

# Name of input file must be given.
filename_ann = config.get('paths', 'filename_ann')
assert filename_ann != '', "Filename of input model not specified."

# Check that simulator choice is valid.
simulator = config.get('simulation', 'simulator')
simulators = config_string_to_set_of_strings(config.get('restrictions',
                                                        'simulators'))
assert simulator in simulators, \
    "Simulator '{}' not supported. Choose from {}".format(simulator,
                                                          simulators)

# Set default path if user did not specify it.
if config.get('paths', 'path_wd') == '':
    config.set('paths', 'path_wd', os.path.dirname(config_filepath))

# Check specified working directory exists.
path_wd = config.get('paths', 'path_wd')
assert os.path.exists(path_wd), \
    "Working directory {} does not exist.".format(path_wd)

# Check that choice of input model library is valid.
model_lib = config.get('input', 'model_lib')
h5_filepath = str(os.path.join(path_wd, filename_ann + '.h5'))
assert os.path.isfile(h5_filepath), \
    "File {} not found.".format(h5_filepath)
json_file = filename_ann + '.json'
if not os.path.isfile(os.path.join(path_wd, json_file)):
    import keras
    import h5py
    from snntoolbox.parsing.utils import get_custom_activations_dict
    # Remove optimizer_weights here, because they may cause the
    # load_model method to fail if the network was trained on a
    # different platform or keras version
    # (see https://github.com/fchollet/keras/issues/4044).
    with h5py.File(h5_filepath, 'a') as f:
        if 'optimizer_weights' in f.keys():
            del f['optimizer_weights']
    # Try loading the model.
    keras.models.load_model(h5_filepath, get_custom_activations_dict())

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
plot_vars = config_string_to_set_of_strings(config.get('restrictions',
                                                       'plot_vars'))
assert all([v in plot_vars for v in plot_var]), \
    "Plot variable(s) {} not understood.".format(
        [v for v in plot_var if v not in plot_vars])
if 'all' in plot_var:
    plot_vars_all = plot_vars.copy()
    plot_vars_all.remove('all')
    config.set('output', 'plot_vars', str(plot_vars_all))

log_var = get_log_keys(config)
log_vars = config_string_to_set_of_strings(config.get('restrictions',
                                                      'log_vars'))
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

with open(os.path.join(log_dir_of_current_run, '.config'), str('w')) as f:
    config.write(f)

"""+++++++++++++++++++++++++++++ PIPELINE +++++++++++++++++++++++++++++++++++"""

# Only needed for GUI.
queue = None

num_to_test = config.getint('simulation', 'num_to_test')

# Instantiate an empty spiking network
spiking_model = SNN(config, queue)

# ____________________________ LOAD DATASET ______________________________ #

normset, testset = get_dataset(config)

parsed_model = None
if config.getboolean('tools', 'parse') and not is_stop(queue):

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
        model_parser.evaluate(config.getint(
            'simulation', 'batch_size'), num_to_test, **testset)

    # Write parsed model to disk
    parsed_model.save(str(
        os.path.join(config.get('paths', 'path_wd'),
                     config.get('paths', 'filename_parsed_model') + '.h5')))

# ______________________________ CONVERT _________________________________ #

if config.getboolean('tools', 'convert') and not is_stop(queue):
    if parsed_model is None:
        from snntoolbox.parsing.model_libs.keras_input_lib import load
        try:
            parsed_model = load(
                config.get('paths', 'path_wd'),
                config.get('paths', 'filename_parsed_model'),
                filepath_custom_objects=config.get(
                    'paths', 'filepath_custom_objects'))['model']
        except FileNotFoundError:
            print("Could not find parsed model {} in path {}. Consider "
                  "setting `parse = True` in your config file.".format(
                    config.get('paths', 'path_wd'),
                    config.get('paths', 'filename_parsed_model')))

    spiking_model.build(parsed_model)

    # Export network in a format specific to the simulator with which it
    # will be tested later.
    spiking_model.save(config.get('paths', 'path_wd'),
                       config.get('paths', 'filename_snn'))

# _______________________________ SIMULATE _______________________________ #

if config.getboolean('tools', 'simulate') and not is_stop(queue):

    # Simulate network
    spiking_model.run(**testset)

    # Clean up
    spiking_model.end_sim()
