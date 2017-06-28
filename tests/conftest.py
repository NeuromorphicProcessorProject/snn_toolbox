# coding=utf-8

"""py.test fixtures with module-scope."""

import os
from importlib import import_module

import pytest


def is_module_installed(mod):
    import sys
    if sys.version_info[0] < 3:
        import pkgutil
        return pkgutil.find_loader(mod) is not None
    else:
        import importlib
        return importlib.util.find_spec(mod) is not None


@pytest.fixture(scope='module')
def _config():
    from snntoolbox.bin.utils import update_setup
    return update_setup(os.path.join(os.path.dirname(__file__),
                                     'configurations', 'config0'))


@pytest.fixture(scope='session')
def _path_wd(tmpdir_factory):
    return tmpdir_factory.mktemp('wd')


@pytest.fixture(scope='session')
def _datapath(_path_wd):
    return _path_wd.mkdir('dataset')


@pytest.fixture(scope='module')
def _dataset(_config):
    from snntoolbox.datasets.utils import get_dataset
    return get_dataset(_config)


@pytest.fixture(scope='module')
def _normset(_dataset):
    return _dataset[0]


@pytest.fixture(scope='module')
def _testset(_dataset):
    return _dataset[1]


def get_input_libs():
    ml = []
    path_wd = os.path.abspath(os.path.join(os.path.dirname(__file__), '..',
                                           'examples', 'models'))
    if is_module_installed('keras'):
        ml.append({'input': {'model_lib': 'keras'},
                   'paths': {'filename_ann': '98.96', 'path_wd':
                             os.path.join(path_wd, 'lenet5', 'keras')},
                   'simulation': {'target_acc': '98.50'}})
    if is_module_installed('caffe'):
        ml.append({'input': {'model_lib': 'caffe'},
                   'paths': {'filename_ann': '96.13', 'path_wd':
                             os.path.join(path_wd, 'lenet5', 'caffe')},
                   'simulation': {'target_acc': '95.10'}})
    if is_module_installed('lasagne'):
        ml.append({'input': {'model_lib': 'lasagne'},
                   'paths': {'filename_ann': '99.02', 'path_wd':
                             os.path.join(path_wd, 'lenet5', 'lasagne')},
                   'simulation': {'target_acc': '98.00'}})
    return ml

_ml = get_input_libs()


@pytest.fixture(scope='module')
def _parsed_model(_config, _normset):
    from snntoolbox.conversion.utils import normalize_parameters
    input_model_and_lib = _input_model_and_lib_single(_config, _ml[0])
    model = input_model_and_lib['input_model']['model']
    model_parser = input_model_and_lib['model_lib'].ModelParser(model, _config)
    model_parser.parse()
    parsed_model = model_parser.build_parsed_model()
    normalize_parameters(parsed_model, _config, **_normset)
    return parsed_model


@pytest.fixture(scope='module')
def _input_model_and_lib_single(_config, p):
    _config.read_dict(p)
    model_lib = import_module('snntoolbox.parsing.model_libs.' +
                              _config['input']['model_lib'] + '_input_lib')
    input_model = model_lib.load(_config['paths']['path_wd'],
                                 _config['paths']['filename_ann'])
    return {'model_lib': model_lib, 'input_model': input_model,
            'target_acc': _config.getfloat('simulation', 'target_acc')}


@pytest.fixture(scope='module', params=_ml)
def _input_model_and_lib(_config, request):
    return _input_model_and_lib_single(_config, request.param)


def get_parameters_for_simtests():
    from snntoolbox.bin.utils import load_config, initialize_simulator

    config_defaults = load_config(os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', 'snntoolbox', 'config_defaults')))

    config_ini = {'simulation': {'simulator': 'INI', 'target_acc': 99.00,
                                 'num_to_test': 200}}
    config_nest = {'simulation': {'simulator': 'nest', 'target_acc': 99.00,
                                  'num_to_test': 2},
                   'input': {'poisson_input': True}}
    config_brian = {'simulation': {'simulator': 'brian', 'target_acc': 99.00,
                                   'num_to_test': 2},
                    'input': {'poisson_input': True}}
    config_neuron = {'simulation': {'simulator': 'Neuron', 'target_acc': 99.00,
                                    'num_to_test': 2},
                     'input': {'poisson_input': True}}
    config_megasim = {'simulation': {'simulator': 'MegaSim', 'batch_size': 1,
                                     'target_acc': 99.00, 'num_to_test': 2},
                      'input': {'poisson_input': True}}
    configs_to_test = [config_ini, config_nest, config_brian, config_neuron,
                       config_megasim]
    _sm = []
    for config_to_test in configs_to_test:
        config_defaults.read_dict(config_to_test)
        try:
            initialize_simulator(config_defaults)
        except (ImportError, KeyError, ValueError):
            continue
        config_defaults['restrictions']['is_installed'] = 'True'
        _sm.append(config_defaults)
    return _sm

sm = get_parameters_for_simtests()


@pytest.fixture(scope='module', params=sm)
def _spiking_model_and_sim(_config, _path_wd, request):
    _config.read_dict(request.param)
    _config.set('paths', 'path_wd', str(_path_wd))
    target_sim = import_module('snntoolbox.simulation.target_simulators.' +
                               _config['simulation']['simulator'] +
                               '_target_sim')
    spiking_model = target_sim.SNN(_config)
    return {'target_sim': target_sim, 'spiking_model': spiking_model,
            'target_acc': request.param['simulation']['target_acc']}


def get_examples():
    example_filepaths = []
    path_wd = os.path.abspath(os.path.join(os.path.dirname(__file__), '..',
                                           'examples', 'models'))
    if is_module_installed('keras'):
        example_filepaths.append(os.path.join(path_wd, 'inceptionV3', 'config'))
    if is_module_installed('lasagne'):
        example_filepaths.append(os.path.join(path_wd, 'binaryconnect',
                                              'config'))
        example_filepaths.append(os.path.join(path_wd, 'binarynet', 'config'))
    return example_filepaths


_example_filepaths = get_examples()


@pytest.fixture(scope='module', params=_example_filepaths)
def _example_filepath(request):
    return request.param
