# coding=utf-8

"""Test configuration of toolbox."""

import os
import pytest
import snntoolbox.config as config
from snntoolbox.core.util import get_root_dir


def get_modules_to_import(root_dir):
    requs = list(open(os.path.join(root_dir, 'requirements.txt')))
    return [d.strip('\n') for d in requs if '#' not in d and d != '\n']

imps = get_modules_to_import(get_root_dir())


@pytest.mark.parametrize('required_module', imps)
def test_imports_from_requirements(required_module):
    import importlib
    assert importlib.import_module(required_module)


in_and_out = [
    ({}, False),
    ({'path_wd': os.path.join(get_root_dir(), 'examples'),
      'dataset_path': os.path.join(get_root_dir(), 'examples', 'dataset'),
      'filename_ann': '83.62'}, True)
    ]


@pytest.mark.parametrize('params, expect_pass', in_and_out)
def test_updating_settings(params, expect_pass):
    if expect_pass:
        assert config.update_setup(params)
    else:
        pytest.raises(AssertionError, config.update_setup, params)


# Specify which simulators are installed:
simulators = [
    ('INI', {}, True),
    ('brian2', {}, False),
    ('MegaSim', {}, True),
    ('brian', {'dt': 1}, False),
    ('nest', {'dt': 1}, False),
    ('Neuron', {'dt': 1}, False)]


@pytest.mark.parametrize('simulator, kwargs, installed', simulators)
def test_initialize_simulator(simulator, kwargs, installed):
    if installed:
        assert config.initialize_simulator(simulator, **kwargs)
    else:
        pytest.raises(ImportError, config.initialize_simulator,
                      simulator=simulator, **kwargs)
