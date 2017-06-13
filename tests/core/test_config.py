# coding=utf-8

"""Test configuration of toolbox."""

import os
import pytest


def get_modules_to_import(root_dir):
    requs = list(open(os.path.join(root_dir, 'requirements.txt')))
    return [d.strip('\n') for d in requs if '#' not in d and d != '\n']

_imps = get_modules_to_import(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


@pytest.mark.parametrize('required_module', _imps)
def test_imports_from_requirements(required_module):
    import importlib
    assert importlib.import_module(required_module)


_in_and_out = [
    ({}, False),
    ({'paths': {'path_wd': os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', '..', 'examples', 'models', 'lenet5',
        'keras')),
      'dataset_path': os.path.abspath(os.path.join(
          os.path.dirname(__file__), '..', '..', 'examples', 'datasets',
          'mnist')),
      'filename_ann': '98.96'}}, True)
    ]


@pytest.mark.parametrize('params, expect_pass', _in_and_out)
def test_updating_settings(params, expect_pass, _path_wd):
    from snntoolbox.config import update_setup
    try:
        import configparser
    except ImportError:
        import ConfigParser as configparser
    config = configparser.ConfigParser()
    config.read_dict(params)
    configpath = os.path.join(str(_path_wd), 'config')
    with open(configpath, 'w') as f:
        config.write(f)
    if expect_pass:
        assert update_setup(configpath)
    else:
        pytest.raises(AssertionError, update_setup, configpath)


# Specify which simulators are installed:
_simulators = [
    ('INI', {}, True),
    ('brian2', {}, False),
    ('MegaSim', {}, True),
    ('brian', {'dt': 1}, False),
    ('nest', {'dt': 1}, False),
    ('Neuron', {'dt': 1}, False)]


@pytest.mark.parametrize('simulator, kwargs, installed', _simulators)
def test_initialize_simulator(simulator, kwargs, installed):
    from snntoolbox.config import initialize_simulator
    if installed:
        assert initialize_simulator(simulator, **kwargs)
    else:
        pytest.raises(ImportError, initialize_simulator,
                      simulator=simulator, **kwargs)
