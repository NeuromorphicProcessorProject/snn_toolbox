# coding=utf-8

"""Test configuration of toolbox."""

import pytest
import snntoolbox.config as config

requs = list(open('../../requirements.txt'))
imps = [d.strip('\n') for d in requs if '#' not in d and d != '\n']


@pytest.mark.parametrize('required_module', imps)
def test_imports_from_requirements(required_module):
    import importlib
    assert importlib.import_module(required_module)


in_and_out = [
    ({}, False),
    ({'path_wd': '../../examples', 'dataset_path': '../../examples/dataset',
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
    ('brian2', {}, True),
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
