# coding=utf-8

"""py.test fixtures with session-scope."""

import os
import pytest
from importlib import import_module


def is_module_installed(mod):
    import sys
    if sys.version_info[0] < 3:
        import pkgutil
        return pkgutil.find_loader(mod) is not None
    else:
        import importlib
        return importlib.util.find_spec(mod) is not None


@pytest.fixture(scope='session')
def settings():
    from snntoolbox.config import settings as s
    return s


@pytest.fixture(scope='session')
def path_wd(tmpdir_factory):
    return tmpdir_factory.mktemp('wd')


@pytest.fixture(scope='session')
def datapath(path_wd):
    return path_wd.mkdir('dataset')


@pytest.fixture(scope='session')
def dataset(settings):
    from snntoolbox.core.util import get_dataset, get_root_dir
    s = settings.copy()
    s['dataset_path'] = os.path.join(get_root_dir(), 'tests', 'models')
    return get_dataset(s)


@pytest.fixture(scope='session')
def evalset(dataset):
    return dataset[0]


@pytest.fixture(scope='session')
def normset(dataset):
    return dataset[1]


@pytest.fixture(scope='session')
def testset(dataset):
    return dataset[2]


def get_input_libs():
    ml = []
    if is_module_installed('keras'):
        ml.append({'model_lib': 'keras', 'filename_ann': '99.32'})
    if is_module_installed('caffe'):
        ml.append({'model_lib': 'caffe', 'filename_ann': '99.00'})
    if is_module_installed('lasagne'):
        ml.append({'model_lib': 'lasagne', 'filename_ann': '99.02'})
    return ml

ml = get_input_libs()


@pytest.fixture(scope='session')
def parsed_model(path_wd, normset):
    from snntoolbox.core.util import parse, normalize_parameters
    input_model_and_lib = input_model_and_lib_single(ml[0])
    parsed_model = parse(input_model_and_lib['input_model']['model'])
    normalize_parameters(parsed_model, path=str(path_wd), **normset)
    return parsed_model


@pytest.fixture(scope='session')
def input_model_and_lib_single(p):
    from snntoolbox.core.util import get_root_dir
    model_lib = import_module('snntoolbox.model_libs.' +
                              p['model_lib'] + '_input_lib')
    path_wd = os.path.join(get_root_dir(), 'tests', 'models', p['model_lib'])
    input_model = model_lib.load_ann(path_wd, p['filename_ann'])
    return {'model_lib': model_lib, 'input_model': input_model,
            'target_acc': float(p['filename_ann'])}


@pytest.fixture(scope='session', params=ml)
def input_model_and_lib(request):
    return input_model_and_lib_single(request.param)


def get_parameters_for_simtests():
    from snntoolbox.config import initialize_simulator
    sm = [{'simulator': 'INI', 'target_acc': 99.00, 'num_to_test': 200}]
    try:
        initialize_simulator('nest', dt=1)
        sm.append({'simulator': 'nest', 'target_acc': 99.00, 'num_to_test': 2,
                   'poisson_input': True})
    except ImportError:
        pass
    try:
        initialize_simulator('brian', dt=1)
        sm.append({'simulator': 'brian', 'target_acc': 99.00, 'num_to_test': 2,
                   'poisson_input': True})
    except ImportError:
        pass
    try:
        initialize_simulator('Neuron', dt=1)
        sm.append({'simulator': 'Neuron', 'target_acc': 99.00, 'num_to_test': 2,
                   'poisson_input': True})
    except ImportError:
        pass
    try:
        import_module('snntoolbox.core.megasim')
        sm.append({'simulator': 'MegaSim', 'target_acc': 99.00,
                   'num_to_test': 2, 'poisson_input': True, 'batch_size': 1})
    except ImportError:
        pass
    return sm

sm = get_parameters_for_simtests()


@pytest.fixture(scope='session', params=sm)
def spiking_model_and_sim(request, settings):
    target_sim = import_module('snntoolbox.target_simulators.' +
                               request.param['simulator'] + '_target_sim')
    settings.update(request.param)
    spiking_model = target_sim.SNN(settings)
    return {'target_sim': target_sim, 'spiking_model': spiking_model,
            'target_acc': request.param['target_acc']}
