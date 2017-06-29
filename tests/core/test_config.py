# coding=utf-8

"""Test configuration of toolbox."""

import os

import pytest

from tests.conftest import sm


def get_modules_to_import():
    return ['future', 'keras']

_imps = get_modules_to_import()


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
    from snntoolbox.bin.utils import update_setup
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


@pytest.mark.parametrize('config', sm)
def test_initialize_simulator(config):
    from snntoolbox.bin.utils import initialize_simulator
    if config.getboolean('restrictions', 'is_installed'):
        assert initialize_simulator(config)
    else:
        pytest.raises(ImportError, initialize_simulator, config=config)
